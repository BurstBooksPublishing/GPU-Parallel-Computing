python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main():
    # Initialize distributed training if launched with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if dist.is_available() and dist.is_nccl_available():
        dist.init_process_group(backend="nccl")

    # Model, dataset, optimizer
    model = models.resnet50(weights=None).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = datasets.FakeData(size=10240, transform=transform)
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    train_loader = DataLoader(
        dataset, batch_size=64, shuffle=(sampler is None),
        num_workers=4, pin_memory=True, sampler=sampler
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

if __name__ == "__main__":
    main()