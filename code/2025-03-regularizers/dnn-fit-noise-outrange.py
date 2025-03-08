#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Demonstrate how a piecewise approximation can be built with linear layers.
import numpy
import torch

def sample_curve(x):
    """Produce a curve for fitting examples."""
    return 2**(-10*(x - 0.5)**2)


# The x and y points along a curve
x_samples = [0.05 * x for x in range(21)]
y_samples = [sample_curve(x) for x in x_samples]
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
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1))
# Results are less predictable without momentum
optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.05)
loss_fn = torch.nn.MSELoss(reduction='sum')

net.train()

x_inputs = torch.tensor(x_samples).view((len(x_samples), 1))
y_targets = torch.tensor(y_samples).view((len(y_samples), 1))

# Train for 4000 steps
for step in range(4000):
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
x_samples = [-1 + 0.025 * x for x in range(121)]
y_samples = [sample_curve(x) for x in x_samples]
prediction = net(torch.tensor(x_samples).view((len(x_samples), 1))).flatten().tolist()
for idx, point in enumerate(zip(x_samples, y_samples)):
    if point[0] < 0 or point[0] > 1 or idx % 2 == 1:
        noise_val = 'none'
    else:
        noise_val = point[1] + noise[(idx - 40)//2]
    print(f"{point[0]}, {point[1]}, {noise_val}, {prediction[idx]}")

