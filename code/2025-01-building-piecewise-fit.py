#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Demonstrate how a piecewise approximation can be built with linear layers.
import functools
import random
import torch

def sample_curve(offset, spread, magnitude, x):
    """This will look like a normal distribution with maximum of the given magnitude."""
    return magnitude * 2**(-(x - offset)**2/spread)

# The basic example of overfitting from "The Little Book of Deep Learning" involves summing several curves into a larger curve
# The author discusses the example with Gaussian kernels, but let's talk about NNs

basis_functions = [functools.partial(sample_curve, offset/4.0, 1, 1) for offset in range(4)]

def correct_output(x):
    return sum([basis(x) for basis in basis_functions])

net = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1))

net.train()

# x values range from 0 to 1
# Let's break it into 4 pieces. Find the slopes of those pieces
stepsize = 0.25
slopes = torch.tensor([(correct_output(x/4.0+stepsize)-correct_output(x/4.0))/stepsize for x in range(4)])
first_value = correct_output(0)

# We will have to do some sign flipping for negative slopes to get through the ReLU
negative_slopes = slopes < 0
abs_slopes = slopes.abs()

with torch.no_grad():
    # The weights will create the slopes of the pieces for the piecewise approximate.
    # To "turn off" a piece, use another weight with the opposite value. That won't be able to get through the ReLU, so copy the weights in the first layer and negate the weights in the second linear layer
    net[0].weight[0:4,0] = abs_slopes
    net[0].weight[4:,0] = abs_slopes
    # These "turn on" the slopes when the bias is cleared (since a ReLU outputs 0 with any negative value)
    net[0].bias[0] = 0. * abs_slopes[0] + first_value
    net[0].bias[1] = -0.25 * abs_slopes[1]
    net[0].bias[2] = -0.5 * abs_slopes[2]
    net[0].bias[3] = -0.75 * abs_slopes[3]
    # To "turn off" the slopes, set the negative slopes to activate when the piece is complete.
    net[0].bias[4] = -0.25 * abs_slopes[0]
    net[0].bias[5] = -0.5 * abs_slopes[1]
    net[0].bias[6] = -0.75 * abs_slopes[2]
    net[0].bias[7] = -1. * abs_slopes[3]

    net[2].weight[0,0:4].fill_(1.)
    net[2].weight[0,4:].fill_(-1.)
    net[2].bias.fill_(0.)

    for s_idx in range(negative_slopes.size(0)):
        if negative_slopes[s_idx]:
            net[2].weight[0,s_idx] *= -1
            net[2].weight[0,s_idx+4] *= -1

net.eval()

print("Piecewise outputs are:")
xs = [0.05 * x for x in range(21)]
x_inputs = torch.tensor(xs).view((len(xs), 1))
outputs = net.forward(x_inputs)
correct_outputs = [[correct_output(x)] for x in xs]
print("Step1, x, desired, predicted")
for i, x in enumerate(xs):
    print("Step1, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + outputs.tolist()[i]]))

################
# Attempt to learn with the same parameters

net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1))
optimizer = torch.optim.SGD(net2.parameters(), lr=0.002, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

net2.train()

xs = [0.05 * x for x in range(21)]
x_inputs = torch.tensor(xs).view((len(xs), 1))
sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

for _ in range(2000):
    optimizer.zero_grad()
    output = net2.forward(x_inputs)
    loss = loss_fn(output, sample_outputs)
    loss.backward()
    optimizer.step()

net2.eval()

print("Learned outputs are:")
model_outputs = net2.forward(x_inputs)
print("Step2, x, desired, predicted")
for i, x in enumerate(xs):
    print("Step2, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + model_outputs.tolist()[i]]))

print("Learned parameters:")
print("Layer 1 weights: {}".format(net2[0].weight))
print("Layer 1 bias: {}".format(net2[0].bias))
print("Layer 3 weights: {}".format(net2[2].weight))
print("Layer 3 bias: {}".format(net2[2].bias))


################
# Pieces of the preset network

print("Step3, x, bias, " + ", ".join(["piece {}".format(i) for i in range(8)]))
with torch.no_grad():
    pieces = net[1](net[0](x_inputs))
    pieces = pieces * net[2].weight[0]
for i, x in enumerate(xs):
    print("Step3, {}, {}, ".format(x, net[2].bias[0].item()) + ", ".join([str(val) for val in pieces[i].tolist()]))

################
# Pieces of the learned network

print("Step4, x, bias, " + ", ".join(["piece {}".format(i) for i in range(8)]))
with torch.no_grad():
    pieces = net2[1](net2[0](x_inputs))
    pieces = pieces * net2[2].weight[0] + net2[2].bias
for i, x in enumerate(xs):
    print("Step4, {}, {}, ".format(x, net2[2].bias[0].item()) + ", ".join([str(val) for val in pieces[i].tolist()]))

# The DNN training process becomes trapped in a local minima.
# Changing any weight or bias alone will lead to a worse result.
# One way to think about the learning process it that it starts with whatever the random parameters are and searches for a solution among them, tuning them to a correct answer while driving any "wrong" parameters to 0. The more parameters we begin with, the more likely we are to find a good solution among them.
# There are some different ways to counter this, but it's a persistent problem.

