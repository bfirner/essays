#!/usr/bin/python3

# This is an analysis of the under and overfitting example in "The Little Book of Deep Learning" by François Fleuret.
# Demonstrate how a piecewise approximation can be built with linear layers.
import math
import torch

def sample_curve(x):
    """Produce a curve for fitting examples."""
    return 2**(-10*(x - 0.5)**2)

# The x and y points along a curve
x_samples = [0.2 * x for x in range(6)]
y_samples = [sample_curve(x) for x in x_samples]

################
# Learn the function and plot the pieces over each step.

# For better repeatability
torch.random.manual_seed(0)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 1))

# Instead of training the model, we will set the parameters so that the output
# intercepts each of the training points.

# This turns off gradient calculations since we aren't do learning.
with torch.no_grad():
    # Initialize all parameters to 0.
    net[0].bias.fill_(0.)
    net[0].weight.fill_(0.)
    net[2].bias.fill_(0.)
    net[2].weight.fill_(0.)

    # Remember the slopes for delta slope calculations
    slopes = [0.]

    # Set all other weight and bias values to handle slopes for the rest of the points
    for i in range(1, len(x_samples)):
        # Calculate the changes in slope required to go from one point to the next
        slope = (y_samples[i]-y_samples[i-1]) / (x_samples[i]-x_samples[i-1])
        slopes.append(slope)
        delta_slope = slopes[-1] - slopes[-2]
        # The weight for the next parameter will be the delta slope
        net[0].weight[i-1,0] = abs(delta_slope)
        # Set the bias value so that the output will be <0 before this training point
        net[0].bias[i-1] = -x_samples[i-1] * abs(delta_slope)
        # In the second linear layer, set the correct sign for the slope
        net[2].weight[0,i-1] = math.copysign(1, delta_slope)
    # Set the bias value to match the first y value of the training points
    net[2].bias[0] = y_samples[0]

#####
## To check for errors:
net.train()
loss_fn = torch.nn.MSELoss(reduction='sum')
x_inputs = torch.tensor(x_samples).view((len(x_samples), 1))
y_targets = torch.tensor(y_samples).view((len(y_samples), 1))
output = net.forward(x_inputs)
loss = loss_fn(output, y_targets)
print(f"Initial loss is {loss}")
optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.05)
# Train for 4000 steps
for step in range(4000):
    #printCurrentModel(step=step, xs=xs, x_inputs=x_inputs, net=net)
    optimizer.zero_grad()
    output = net.forward(x_inputs)
    loss = loss_fn(output, y_targets)
    loss.backward()
    optimizer.step()
print(f"Final loss is {loss}")
#####

net.eval()

# Print out the samples and our predictions
print("x, y samples, prediction")
# Also plot some extra points to see how the fit generalizes between the training points
x_samples = [0.025 * x for x in range(41)]
y_samples = [sample_curve(x) for x in x_samples]
prediction = net(torch.tensor(x_samples).view((len(x_samples), 1))).flatten().tolist()
for idx, point in enumerate(zip(x_samples, y_samples)):
    print(f"{point[0]}, {point[1]}, {prediction[idx]}")
