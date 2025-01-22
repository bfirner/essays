#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Demonstrate how a piecewise approximation can be built with linear layers.
import argparse
import functools
import math
import random
import torch

parser = argparse.ArgumentParser(description='Some experiments with normalization.')

parser.add_argument(
    '--test',
    type=int,
    required=True,
    help='Which test to perform')

args = parser.parse_args()

################
# Functions

def sample_curve(offset, spread, magnitude, x):
    """This will look like a normal distribution with maximum of the given magnitude."""
    return magnitude * 2**(-(x - offset)**2/spread)

# The basic example of overfitting from "The Little Book of Deep Learning" involves summing several curves into a larger curve
# The author discusses the example with Gaussian kernels, but let's talk about NNs

basis_functions = [functools.partial(sample_curve, -2 + offset, 1, 1) for offset in range(5)]

def correct_output(x):
    return sum([basis(x) for basis in basis_functions])

def getLogStrings(step, xs, x_inputs, net, prepend):
    # Get the current state of the model outputs and put plottable strings into a array
    stat_strings = []
    with torch.no_grad():
        num_pieces = net[-1].weight.size(1)
        stat_strings.append(f"{prepend}{step}, x, Target, Model, bias, " + ", ".join(["piece {}".format(i) for i in range(num_pieces)]))
        model_outputs = net.forward(x_inputs)
        pieces = net[0](x_inputs)
        # Not all versions of the network have the nonlinearity in the second layer,
        # so just loop through the interior layers.
        for comp_idx in range(1, len(net)):
            pieces = net[comp_idx](pieces)
        pieces = pieces * net[-1].weight[0]
        for i, x in enumerate(xs):
            stat_strings.append(f"{prepend}{step}, {x}, {correct_output(x)}, {model_outputs.tolist()[i][0]}, {net[-1].bias[0].item()}, " + ", ".join([str(val) for val in pieces[i].tolist()]))
    return stat_strings

def printCurrentModel(step, xs, x_inputs, net, prepend="Step"):
    # Print out the current state of the model outputs
    for line in getLogStrings(step, xs, x_inputs, net, prepend):
        print(line)

################
# Big model

if args.test == 1:

    final_loss = 1
    attempt = 0

    while final_loss > 0.04 and attempt < 100:
        attempt_strs = []
        print(f"attempt {attempt}")
        attempt = attempt + 1

        torch.random.manual_seed(attempt)

        # Typical neural network with lots of parameters.
        net = torch.nn.Sequential(
                torch.nn.Linear(1, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.0)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        net.train()

        # First, let's just fit to 60 points from -2.5 to 1 and from 1 to 2.5.
        xs = sorted([-2.5 + 1.5*random.random() for _ in range(30)] + [1 + 1.5*random.random() for _ in range(30)])
        x_inputs = torch.tensor(xs).view((len(xs), 1))
        sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

        for step in range(1000):
            optimizer.zero_grad()
            output = net.forward(x_inputs)
            loss = loss_fn(output, sample_outputs)
            loss.backward()
            optimizer.step()
        final_loss = loss
    print(f"Final loss is {final_loss} after {attempt} attempts")

    net.eval()

    # Print out the final result across the full range
    xs = [-2.5 + 0.05 * x for x in range(101)]
    x_inputs = torch.tensor(xs).view((len(xs), 1))
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="bigmodel")

################
# Big model

if args.test == 2:

    final_loss = 1
    attempt = 0

    while final_loss > 0.04 and attempt < 100:
        attempt_strs = []
        print(f"attempt {attempt}")
        attempt = attempt + 1

        torch.random.manual_seed(attempt)

        # Typical neural network with lots of parameters.
        net = torch.nn.Sequential(
                torch.nn.Linear(1, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.0)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        net.train()

        # First, let's just fit to 60 points from -2.5 to 1, -0.1 to 0.1, and from 1 to 2.5.
        xs = sorted([-2.5 + 1.5*random.random() for _ in range(30)] + [-0.1 + 0.2*random.random() for _ in range(1)] + [1 + 1.5*random.random() for _ in range(30)])
        x_inputs = torch.tensor(xs).view((len(xs), 1))
        sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

        for step in range(1000):
            optimizer.zero_grad()
            output = net.forward(x_inputs)
            loss = loss_fn(output, sample_outputs)
            loss.backward()
            optimizer.step()
        final_loss = loss
    print(f"Final loss is {final_loss} after {attempt} attempts")

    net.eval()

    # Print out the final result across the full range
    xs = [-2.5 + 0.05 * x for x in range(101)]
    x_inputs = torch.tensor(xs).view((len(xs), 1))
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="bigmodel")

################
# Human linear fit

if args.test == 3:

    # Our sorted samples with 9 elements
    xs = sorted([-2.5 + 1.5*random.random() for _ in range(4)] + [-0.1 + 0.2*random.random() for _ in range(1)] + [1 + 1.5*random.random() for _ in range(4)])
    sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

    # Our DNN with 16 weights in each linear layer.
    net = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1))

    with torch.no_grad():
        net[0].bias.fill_(0.)
        net[0].weight.fill_(0.)
        net[2].bias.fill_(0.)
        net[2].weight.fill_(0.)

        # Remember the slopes for delta slope calculations
        slopes = [0.]

        # Set all other weight and bias values to handle slopes for the rest of the points
        for i in range(1, len(xs)):
            slope = (sample_outputs[i]-sample_outputs[i-1]) / (xs[i]-xs[i-1])
            slopes.append(slope)
            delta_slope = slopes[-1] - slopes[-2]
            net[0].weight[i-1,0] = abs(delta_slope)
            net[0].bias[i-1] = -xs[i-1] * abs(delta_slope)
            net[2].weight[0,i-1] = math.copysign(1, delta_slope)
        net[2].bias[0] = sample_outputs[0]

    net.eval()

    # Print out the final result across the full range
    xs = [-2.5 + 0.05 * x for x in range(101)]
    x_inputs = torch.tensor(xs).view((len(xs), 1))
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="humanmodel")

################
# Human linear fit with bad points

if args.test == 4:

    # Our sorted samples with 9 elements
    xs = sorted([-2.5 + 1.5*random.random() for _ in range(4)] + [-0.1 + 0.2*random.random() for _ in range(1)] + [1 + 1.5*random.random() for _ in range(4)])
    sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

    # Our DNN with 16 weights in each linear layer.
    net = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1))

    with torch.no_grad():
        net[0].bias.fill_(0.)
        net[0].weight.fill_(0.)
        net[2].bias.fill_(0.)
        net[2].weight.fill_(0.)

        # Remember the slopes for delta slope calculations
        slopes = [0.]

        # Set all other weight and bias values to handle slopes for the rest of the points
        for i in range(1, len(xs)):
            slope = (sample_outputs[i]-sample_outputs[i-1]) / (xs[i]-xs[i-1])
            slopes.append(slope)
            delta_slope = slopes[-1] - slopes[-2]
            net[0].weight[i-1,0] = abs(delta_slope)
            net[0].bias[i-1] = -xs[i-1] * abs(delta_slope)
            net[2].weight[0,i-1] = math.copysign(1, delta_slope)
        net[2].bias[0] = sample_outputs[0]

        # Add in two "bad" neurons to the linear layers
        target_slope = (sample_outputs[5] - sample_outputs[4]) - (xs[5] - xs[4])
        net[0].weight[8,0] = 1
        net[0].weight[9,0] = 2
        # The first one "turns on" after the first target point
        net[0].bias[8] = -xs[4]
        # The second one "turns on" halfway to the second
        net[0].bias[9] = 2 * -(xs[4] + (xs[5]-xs[4])/2.0)
        net[2].weight[0,8] = 1
        net[2].weight[0,9] = -1
        # The outputs of the two neurons cancel when x = target_xs[1]
        # To prevent adding error into the network output we need to add a positive slope to the network at target_xs[1]
        new_slope = slopes[6] + 1
        new_delta = new_slope - slopes[5]
        net[0].weight[5,0] = abs(new_delta)
        net[0].bias[5] = -xs[5] * abs(new_delta)
        net[2].weight[0,5] = math.copysign(1, new_delta)

    net.eval()

    loss_fn = torch.nn.MSELoss(reduction='sum')
    with torch.no_grad():
        output = net.forward(torch.tensor(xs).view((len(xs), 1)))
        loss = loss_fn(output, sample_outputs)
        print(f"Loss is {loss}")


    # Print out the final result across the full range
    xs = [-2.5 + 0.05 * x for x in range(101)]
    x_inputs = torch.tensor(xs).view((len(xs), 1))
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="humanmodel_bad")


################
# Learn the function and plot the pieces over each step.

if args.test == 10:

    final_loss = 1
    attempt = 0

    while final_loss > 0.01 and attempt < 100:
        attempt_strs = []
        print(f"attempt {attempt}")
        attempt = attempt + 1

        torch.random.manual_seed(attempt)

        net = torch.nn.Sequential(
                torch.nn.Linear(1, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.0)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        net.train()

        xs = [0.05 * x for x in range(21)]
        x_inputs = torch.tensor(xs).view((len(xs), 1))
        sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

        for step in range(1000):
            attempt_strs = attempt_strs + getLogStrings(step=step, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear")
            optimizer.zero_grad()
            output = net.forward(x_inputs)
            loss = loss_fn(output, sample_outputs)
            loss.backward()
            optimizer.step()
        final_loss = loss
    print(f"Final loss is {final_loss} after {attempt} attempts")

    net.eval()

    for line in attempt_strs:
        print(line)
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear")

################
# Learn the function and plot the pieces over each step.

if args.test == 10:

    final_loss = 1
    attempt = 0

    while final_loss > 0.01 and attempt < 100:
        attempt_strs = []
        print(f"attempt {attempt}")
        attempt = attempt + 1

        torch.random.manual_seed(attempt)

        net = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.Dropout1d(p=0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.0)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        net.train()

        xs = [0.05 * x for x in range(21)]
        x_inputs = torch.tensor(xs).view((len(xs), 1))
        sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

        for step in range(1000):
            attempt_strs = attempt_strs + getLogStrings(step=step, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear_dropout")
            optimizer.zero_grad()
            output = net.forward(x_inputs)
            loss = loss_fn(output, sample_outputs)
            loss.backward()
            optimizer.step()
        final_loss = loss
    print(f"Final loss is {final_loss} after {attempt} attempts")

    net.eval()

    for line in attempt_strs:
        print(line)
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear_dropout")

################
# Learn the function and plot the pieces over each step.

if args.test == 10:
    final_loss = 1
    attempt = 0

    while final_loss > 0.01 and attempt < 100:
        attempt_strs = []
        print(f"attempt {attempt}")
        attempt = attempt + 1

        torch.random.manual_seed(attempt)

        net = torch.nn.Sequential(
                torch.nn.Linear(1, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.0, weight_decay=0.01)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        net.train()

        xs = [0.05 * x for x in range(21)]
        x_inputs = torch.tensor(xs).view((len(xs), 1))
        sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

        for step in range(1000):
            attempt_strs = attempt_strs + getLogStrings(step=step, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear_decay")
            optimizer.zero_grad()
            output = net.forward(x_inputs)
            loss = loss_fn(output, sample_outputs)
            loss.backward()
            optimizer.step()
        final_loss = loss
    print(f"Final loss is {final_loss} after {attempt} attempts")

    net.eval()

    for line in attempt_strs:
        print(line)
    printCurrentModel(step=1000, xs=xs, x_inputs=x_inputs, net=net, prepend="16linear_decay")
