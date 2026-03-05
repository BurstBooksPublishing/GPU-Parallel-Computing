python
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def init_distributed(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class ShardedLinear(nn.Module):
    """
    Column-wise sharded linear layer.
    Each rank owns `out_features // world_size` output channels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if out_features % self.world_size:
            raise ValueError("out_features must be divisible by world_size")

        self.out_per_rank = out_features // self.world_size
        self.device = torch.device("cuda", self.rank)

        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features, device=self.device)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank, device=self.device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul
        out = x @ self.weight.t()
        if self.bias is not None:
            out += self.bias

        # All-gather along feature dimension
        gathered = [torch.empty_like(out) for _ in range(self.world_size)]
        dist.all_gather(gathered, out, group=self.group)
        return torch.cat(gathered, dim=1)