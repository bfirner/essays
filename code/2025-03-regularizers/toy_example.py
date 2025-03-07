#!/usr/bin/python3

# Toy example
import torch

net = torch.nn.Sequential(
        # 2 inputs, 1 output
        torch.nn.Linear(2, 1),
        torch.nn.ReLU())

# We are directly changing the model parameters, so we need to tell PyTorch
# that we don't treat this as learning
with torch.no_grad():
    # There is one bias value in the first layer, since there is one output
    net[0].bias[0] = -1

    # There are two weights in the first layer, for the two inputs
    net[0].weight[0].fill_(0.5)

    # The network performs the operation f(a,b) = ReLU(-1 + 0.5a + 0.5b)
    print(f"f(1., 8.) = {net.forward(torch.tensor([1., 8.]))}")
    print(f"f(0., 0.) = {net.forward(torch.tensor([0., 0.]))}")
    print(f"f(4., -2.) = {net.forward(torch.tensor([4., -2.]))}")
    print(f"f(3., 3.) = {net.forward(torch.tensor([3., 3.]))}")
