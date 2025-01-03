#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Inputs are normal curves, f_i(x)=y
# Output is a different target curve, f_t(x)=y
# DNN maps from all f_i to f_t, but training samples may be sparse.

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


# First, let's just fit to 20 points from 0-0.5 and 0.75-1
xs = sorted([random.random()/2 for _ in range(10)] + [0.75+random.random()/4 for _ in range(10)])
sample_inputs = torch.tensor([[basis(x) for basis in basis_functions] for x in xs])
sample_outputs = torch.tensor([[correct_output(x)] for x in xs])

net = torch.nn.Sequential(
        torch.nn.Linear(4, 60),
        torch.nn.Linear(60, 1))
optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

net.train()

for _ in range(2000):
    optimizer.zero_grad()
    output = net.forward(sample_inputs)
    loss = loss_fn(output, sample_outputs)
    loss.backward()
    optimizer.step()

print("Final loss {}".format(loss))
net.eval()
final_outputs = net.forward(sample_inputs)

print("The final outputs are:")
print("Step1, x, input1, input2, input3, input4, desired, predicted")
for i, x in enumerate(xs):
    print("Step1, " + ", ".join([str(val) for val in [x] + sample_inputs[i].tolist() + sample_outputs.tolist()[i] + final_outputs.tolist()[i]]))

print("Untrained outputs are:")
untrained_xs = [0.05 * x for x in range(21)]
untrained_inputs = torch.tensor([[basis(x) for basis in basis_functions] for x in untrained_xs])
untrained_outputs = net.forward(untrained_inputs)
correct_outputs = [[correct_output(x)] for x in untrained_xs]
print("Step2, x, desired, predicted")
for i, x in enumerate(untrained_xs):
    print("Step2, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + untrained_outputs.tolist()[i]]))


# With such a simple function there won't be overfitting if the source signal are provided.
# How about only providing x?
print("########################################")
print("X input results.")

net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 60),
        torch.nn.ReLU(),
        torch.nn.Linear(60, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1))
optimizer = torch.optim.SGD(net2.parameters(), lr=0.002, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

net.train()

x_inputs = torch.tensor(xs).view((len(xs), 1))
for _ in range(2000):
    optimizer.zero_grad()
    output = net2.forward(x_inputs)
    loss = loss_fn(output, sample_outputs)
    loss.backward()
    optimizer.step()

print("Final loss {}".format(loss))
net2.eval()
final_outputs = net2.forward(x_inputs)

print("The final outputs are:")
print("Step3, x, desired, predicted")
for i, x in enumerate(xs):
    print("Step3, " + ", ".join([str(val) for val in [x] + sample_outputs.tolist()[i] + final_outputs.tolist()[i]]))

print("Untrained outputs are:")
untrained_xs = [0.05 * x for x in range(21)]
untrained_inputs = torch.tensor(untrained_xs).view((len(untrained_xs), 1))
untrained_outputs = net2.forward(untrained_inputs)
correct_outputs = [[correct_output(x)] for x in untrained_xs]
print("Step4, x, desired, predicted")
for i, x in enumerate(untrained_xs):
    print("Step4, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + untrained_outputs.tolist()[i]]))



print("########################################")
print("With noise and sum reduction.")

net3 = torch.nn.Sequential(
        torch.nn.Linear(4, 60),
        torch.nn.Linear(60, 1))
optimizer = torch.optim.SGD(net3.parameters(), lr=0.001, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

net3.train()

for _ in range(2000):
    # A small amount of noise. Divide by 10 to make a stddev of 0.1
    noise = torch.randn(sample_outputs.size()) / 10
    optimizer.zero_grad()
    output = net3.forward(sample_inputs)
    loss = loss_fn(output, sample_outputs + noise)
    loss.backward()
    optimizer.step()

print("Final loss {}".format(loss))
net3.eval()
final_outputs = net3.forward(sample_inputs)

print("The final outputs are:")
print("Step5, x, input1, input2, input3, input4, desired, predicted")
for i, x in enumerate(xs):
    print("Step5, " + ", ".join([str(val) for val in [x] + sample_inputs[i].tolist() + sample_outputs.tolist()[i] + final_outputs.tolist()[i]]))

print("Untrained outputs are:")
untrained_xs = [0.05 * x for x in range(21)]
untrained_inputs = torch.tensor([[basis(x) for basis in basis_functions] for x in untrained_xs])
untrained_outputs = net3.forward(untrained_inputs)
correct_outputs = [[correct_output(x)] for x in untrained_xs]
print("Step6, x, desired, predicted")
for i, x in enumerate(untrained_xs):
    print("Step6, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + untrained_outputs.tolist()[i]]))

print("########################################")
print("With noise and mean reduction.")
# FIXME It seems like the difference is just that the learning rate is 20 times smaller with 'mean' rather than 'sum'.

net4 = torch.nn.Sequential(
        torch.nn.Linear(4, 60),
        torch.nn.Linear(60, 1))
optimizer = torch.optim.SGD(net4.parameters(), lr=20*0.001, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='mean')

net4.train()

for _ in range(2000):
    # A small amount of noise. Divide by 10 to make a stddev of 0.1
    noise = torch.randn(sample_outputs.size()) / 10
    optimizer.zero_grad()
    output = net4.forward(sample_inputs)
    loss = loss_fn(output, sample_outputs + noise)
    loss.backward()
    optimizer.step()

print("Final loss {}".format(loss))
net4.eval()
final_outputs = net4.forward(sample_inputs)

print("The final outputs are:")
print("Step7, x, input1, input2, input3, input4, desired, predicted")
for i, x in enumerate(xs):
    print("Step7, " + ", ".join([str(val) for val in [x] + sample_inputs[i].tolist() + sample_outputs.tolist()[i] + final_outputs.tolist()[i]]))

print("Untrained outputs are:")
untrained_xs = [0.05 * x for x in range(21)]
untrained_inputs = torch.tensor([[basis(x) for basis in basis_functions] for x in untrained_xs])
untrained_outputs = net4.forward(untrained_inputs)
correct_outputs = [[correct_output(x)] for x in untrained_xs]
print("Step8, x, desired, predicted")
for i, x in enumerate(untrained_xs):
    print("Step8, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + untrained_outputs.tolist()[i]]))

# How about with a smaller size for the piecewise fitting?
print("########################################")
print("X input results.")

net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1))
optimizer = torch.optim.SGD(net2.parameters(), lr=0.002, momentum=0.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

net.train()

x_inputs = torch.tensor(xs).view((len(xs), 1))
for _ in range(2000):
    optimizer.zero_grad()
    output = net2.forward(x_inputs)
    loss = loss_fn(output, sample_outputs)
    loss.backward()
    optimizer.step()

print("Final loss {}".format(loss))
net2.eval()
final_outputs = net2.forward(x_inputs)

print("The final outputs are:")
print("Step9, x, desired, predicted")
for i, x in enumerate(xs):
    print("Step9, " + ", ".join([str(val) for val in [x] + sample_outputs.tolist()[i] + final_outputs.tolist()[i]]))

print("Untrained outputs are:")
untrained_xs = [0.05 * x for x in range(21)]
untrained_inputs = torch.tensor(untrained_xs).view((len(untrained_xs), 1))
untrained_outputs = net2.forward(untrained_inputs)
correct_outputs = [[correct_output(x)] for x in untrained_xs]
print("Step10, x, desired, predicted")
for i, x in enumerate(untrained_xs):
    print("Step10, " + ", ".join([str(val) for val in [x] + correct_outputs[i] + untrained_outputs.tolist()[i]]))

# Plot the curves
# Pipe the output into regularizers-example-results.dat and run > gnuplot plot_regularizers.gp

