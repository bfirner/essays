#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Demonstrate how a piecewise approximation can be built with linear layers.
import functools
import numpy
import torch

def sample_curve(offset, spread, magnitude, x):
    """Produce a curve that looks like the overfitting example in "The Little Book of Deep Learning" by François Fleuret."""
    return magnitude * 2**(-(x - offset)**2/spread)


# The x and y points along a curve
x_samples = [0.05 * x for x in range(21)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
noise_generator = numpy.random.default_rng()
noise = numpy.random.standard_normal(len(y_samples)) * 0.05

################
# Learn the function and plot the pieces over each step.

# For better repeatability
torch.random.manual_seed(0)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1))
# Results are less predictable without momentum
optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.05, weight_decay=0.01)
loss_fn = torch.nn.MSELoss(reduction='sum')

net.train()

x_inputs = torch.tensor(x_samples).view((len(x_samples), 1))
y_targets = torch.tensor(y_samples).view((len(y_samples), 1))

# Train for 4000 steps
for step in range(4000):
    #printCurrentModel(step=step, xs=xs, x_inputs=x_inputs, net=net)
    optimizer.zero_grad()
    output = net.forward(x_inputs)
    loss = loss_fn(output, y_targets)
    loss.backward()
    optimizer.step()
    # Note: We could stop early if we achieve good enough results
    # There is no harm is training for longer
    # if loss < 0.005:
    #     break

net.eval()

# Print out the samples and our predictions
print("x, y samples, y noise, prediction")
# Also plot some extra points to see how the fit generalizes between the training points
x_samples = [0.025 * x for x in range(41)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
prediction = net(torch.tensor(x_samples).view((len(x_samples), 1))).flatten().tolist()
for idx, point in enumerate(zip(x_samples, y_samples)):
    if idx % 2 == 0:
        print(f"{point[0]}, {point[1]}, {point[1] + noise[idx//2]}, {prediction[idx]}")
    else:
        print(f"{point[0]}, {point[1]}, none, {prediction[idx]}")

