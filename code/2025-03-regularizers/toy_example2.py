#!/usr/bin/python3

# Toy example
import torch

net = torch.nn.Sequential(
        # 1 inputs, 2 output
        torch.nn.Linear(1, 2),
        torch.nn.ReLU(),
        # 2 inputs, 1 outputs
        torch.nn.Linear(2, 1))

# We are directly changing the model parameters, so we need to tell PyTorch
# that we don't treat this as learning
with torch.no_grad():
    # There are two bias values in the first layer, since there are two outputs
    net[0].bias[0] = 1
    net[0].bias[1] = -1

    # There are two weights in the first layer, for the two outputs
    # The first index of a linear layer's weights is the output number,
    # the second is the input number.
    net[0].weight[0,0] = 1
    net[0].weight[1,0] = 2

    # The first two layers of the network have two outputs:
    #  f_1(x) = ReLU(1 + x)
    #  f_2(x) = ReLU(-1 + 2x)

    # There is one bias value in the third layer, for the one output.
    net[2].bias[0] = 0.25

    # There are two weight values in the first layer, one for each input.
    net[2].weight[0,0] = 0.75
    net[2].weight[0,1] = -0.75

    # The network performs g(x) = 0.25 + 0.75f_1(x) - 0.75f_2(x)
    # g(x) = 0.25 + 0.75*RelU(1 + x) - 0.75*ReLU(-1 + 2x)

    for x in [-1 + inc*0.25 for inc in range(9)]:
        print(f"g({x}) = {net.forward(torch.tensor([x]))}")
