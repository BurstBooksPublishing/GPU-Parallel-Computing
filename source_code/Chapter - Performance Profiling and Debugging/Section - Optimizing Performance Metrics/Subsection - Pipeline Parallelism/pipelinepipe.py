python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader, TensorDataset

def main():
    # deterministic run
    torch.manual_seed(42)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need ≥2 GPUs")

    # local rank for single-node; keep first GPU free for master
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # partition-aware model
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = get_loader(batch_size=32)

    model.train()
    for epoch in range(3):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).local_value()          # Pipe returns a `Future`
            loss = criterion(out.view(-1, out.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

# ------------------------------------------------------------------
def build_model():
    d_model, nhead, num_layers, dim_ff = 1024, 8, 12, 4096
    encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff)
    full = nn.Sequential(
        nn.Linear(1024, d_model),
        *(encoder for _ in range(num_layers)),
        nn.Linear(d_model, 1000)
    )
    # split evenly across GPUs
    stages = nn.ModuleList()
    per = (len(full) + 1) // 2
    for i in range(2):
        sub = full[i*per : (i+1)*per]
        stages.append(nn.Sequential(*sub).to(i))
    return Pipe(nn.Sequential(*stages), chunks=4, devices=[0, 1])

# ------------------------------------------------------------------
def get_loader(batch_size):
    x = torch.randn(1024, 10, 1024)
    y = torch.randint(0, 1000, (1024,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    main()