python
import os
import argparse
import socket
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def setup(rank, world_size, backend='nccl'):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_loop(rank, world_size, args):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)  # unique seed per worker
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.FakeData(size=100000, image_size=(3,224,224), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = torch.tensor(0.0, device='cuda')
        for step, (images, targets) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            running_loss += loss.detach()

            if step % args.accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)
        if rank == 0:
            logging.info(f"Epoch {epoch}: loss={running_loss.item()/step:.4f}")
        scheduler.step()

    if rank == 0:
        torch.save(model.module.state_dict(), "final.pt")
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()
    mp.spawn(train_loop, args=(args.world_size, args), nprocs=args.world_size, join=True)