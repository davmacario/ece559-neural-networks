import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc1(x)


class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(x)


def run(rank, world_size):
    setup(rank, world_size)

    # Partition the model
    if rank == 0:
        model_part1 = ModelPart1().to(rank)
        optimizer1 = optim.SGD(model_part1.parameters(), lr=0.01)
    else:
        model_part2 = ModelPart2().to(rank)
        optimizer2 = optim.SGD(model_part2.parameters(), lr=0.01)

    # Dummy data
    input_data = torch.randn(5, 10).to(rank) if rank == 0 else None
    target = torch.randn(5, 1).to(rank) if rank == 1 else None

    # Forward pass
    if rank == 0:
        activation = model_part1(input_data)
        dist.send(tensor=activation, dst=1)
    else:
        activation = torch.zeros(5, 20).to(rank)
        dist.recv(tensor=activation, src=0)
        output = model_part2(activation)

    # Compute loss
    if rank == 1:
        criterion = nn.MSELoss()
        loss = criterion(output, target)

    # Backward pass
    if rank == 1:
        loss.backward()
        dist.send(tensor=activation.grad, dst=0)
        optimizer2.step()
    else:
        activation_grad = torch.zeros(5, 20).to(rank)
        dist.recv(tensor=activation_grad, src=1)
        activation.backward(activation_grad)
        optimizer1.step()

    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    # run(rank=int(sys.argv[1]), world_size=world_size)
