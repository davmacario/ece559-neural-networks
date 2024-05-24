import threading
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act = F.relu

    def forward(self, x):
        return self.act(self.fc1(x))


class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(x)


# -------------------------------------------------------------------------------------

forw_queue = deque([])
backw_queue = deque([])

fq_nonempty = threading.Event()
fq_nonempty.clear()
bq_nonempty = threading.Event()
bq_nonempty.clear()

EPOCHS = 10


def rank_1(model, input):
    """
    Function for the first node (input)
    """
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for ep in range(EPOCHS):
        activation = model(input)
        activation.retain_grad()
        # Place msg in queue
        forw_queue.append(activation.clone())
        fq_nonempty.set()
        print("[THR1]: Placed activation in queue")

        # Wait for backpropagation
        bq_nonempty.wait()
        activation_grad = backw_queue.popleft()
        bq_nonempty.clear()
        print("[THR1]: Retrieved gradient")
        activation.backward(activation_grad, retain_graph=True)
        optimizer.step()
        print(f"[THR1]: Training iteration {ep + 1} complete!")


def rank_2(model, targets):
    """
    Function for the last node (output)
    """
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    criterion = nn.MSELoss()

    ep = 0  # Should be agnostic of iterations
    while ep < EPOCHS:
        # Wait for the message to arrive
        fq_nonempty.wait()
        activation = forw_queue.popleft()
        fq_nonempty.clear()
        print("[THR2]: Retrieved activation")
        activation.retain_grad()
        output = model(activation)
        loss = criterion(output, targets)
        print(f"[THR2]: Found output and loss = {loss:.4f}; starting backpropagation")

        # Backprop
        loss.backward(retain_graph=True)
        backw_queue.append(activation.grad)
        bq_nonempty.set()
        print("[THR2]: Placed gradient in queue")
        optimizer.step()
        print(f"[THR2]: Training iteration {ep + 1} complete!")
        ep += 1


def run():
    model1 = ModelPart1()
    model2 = ModelPart2()

    # Dummy data
    input_data = torch.randn(1000, 10)
    targets = torch.randn(1000, 1)

    thr_1 = threading.Thread(target=rank_1, args=(model1, input_data))
    thr_2 = threading.Thread(target=rank_2, args=(model2, targets))

    thr_1.start()
    thr_2.start()

    thr_1.join()
    thr_2.join()

if __name__ == "__main__":
    run()
