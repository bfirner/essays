---
title: Regularizers in Neural Networks
---

<!--
Abstract:

You may have heard that overfitting is a problem in machine learning. You may have even heard that regularization fixes the problem. But what is regularization?
Regularization techniques predate modern machine learning, including deep neural networks. Although deep neural networks are surprisingly robust to overfitting, regularization is still used during neural network training. In this talk, we will look at overfitting and regularization in neural networks. How do they resist overfitting? What techniques can we use to regularize neural networks? What are some of the problems that regularization solves?
-->

# Regularizers in Neural Networks

---

## What is a Regularizer?

* Regularizers "simplify" models
  * Reduce "overfitting" to noise in training
  * Improve generalization
* They've long been a part of statistical methods

---

## Least Squares Overfitting

```python
import numpy

def sample_curve(offset, spread, magnitude, x):
    """Produce a curve that looks like the overfitting example in "The Little Book of Deep Learning" by François Fleuret."""
    return magnitude * 2**(-(x - offset)**2/spread)


# The x and y points along a curve
x_samples = [0.05 * x for x in range(21)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]

# The perfect solution to a noiseless set of points.
# We will solve with a as many coefficients as samples
A = numpy.vander(x_samples, N=20, increasing=True)
coef = numpy.linalg.lstsq(A, y_samples, rcond=-1)[0]
# Print out the samples and our fit line
print("x, y samples, fit")
# Also plot some extra points to see how the fit generalizes between the training points
x_samples = [0.025 * x for x in range(41)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
for idx, point in enumerate(zip(x_samples, y_samples)):
    prediction = sum([c * point[0]**i for i, c in enumerate(coef)])
    print(f"{point[0]}, {point[1]}, {prediction}")
```

---

## Without Noise

<!--
* The canonical example of a model with too many parameters "overfitting" to noise
-->

![](./figures/least-squares-no-noise.png)

---

## Adding Noise

```python
noise_generator = numpy.random.default_rng()
noise = numpy.random.standard_normal(len(y_samples)) * 0.05

# The perfect solution to a noiseless set of points.
# We will solve with a as many coefficients as samples
A = numpy.vander(x_samples, N=5, increasing=True)
coef = numpy.linalg.lstsq(A, y_samples + noise, rcond=-1)[0]
A_over = numpy.vander(x_samples, N=20, increasing=True)
coef_over = numpy.linalg.lstsq(A_over, y_samples + noise, rcond=-1)[0]
# Print out the samples and our fit line
print("x, y samples, y noise, fit, overfit")
# Also plot some extra points to see how the fit generalizes between the training points
x_samples = [0.025 * x for x in range(41)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
for idx, point in enumerate(zip(x_samples, y_samples)):
    prediction = sum([c * point[0]**i for i, c in enumerate(coef)])
    overfit_prediction = sum([c * point[0]**i for i, c in enumerate(coef_over)])
    if idx % 2 == 0:
        print(f"{point[0]}, {point[1]}, {point[1] + noise[idx//2]}, {prediction}, {overfit_prediction}")
    else:
        print(f"{point[0]}, {point[1]}, none, {prediction}, {overfit_prediction}")
```

---

## Least Squares with Noise

![](./figures/least-squares-with-noise.png)

The "overfit" line comes from a solution with as many parameters as sample points.\
The "fit" line has a quarter as many.

---

## Regularizers in Neural Networks

* Parameters vastly outnumber the problem dimension
* Modern groups *do not* try to use smaller models
  * Instead, we attempt to use the largest model possible
* Why?
  * Unexpectedly, larger models generalize better than smaller models
\
\
\
Further reading: [The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233)
<!-- Make a drop down with some images from Anna's paper -->

---

## Really?

Let's take an example

```python
# For better repeatability
torch.random.manual_seed(0)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 1000),
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
x_samples = [0.025 * x for x in range(41)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
prediction = net(torch.tensor(x_samples).view((len(x_samples), 1))).flatten().tolist()
for idx, point in enumerate(zip(x_samples, y_samples)):
    if idx % 2 == 0:
        print(f"{point[0]}, {point[1]}, {point[1] + noise[idx//2]}, {prediction[idx]}")
    else:
        print(f"{point[0]}, {point[1]}, none, {prediction[idx]}")
```


<!-- Code for oversized DNN and results -->

---

## Least Squares with Noise

<!--
* The canonical example of a model with too many parameters "overfitting" to noise
-->

![](./figures/dnn-with-noise.png)

Magic!

---

## Why Does This Work?

* Short answer: *Gradient Descent* is *magic*
  * Longer answer is that success will vary:
    * with the kind of noise
    * with the problem
  * Here, the local minima resists moving into a tortured function
    * Local minima: fancy way to describe where the NN parameters get "stuck"
    * The output is a piecewise linear fit with 1000 pieces, which is naturally smooth
<!-- Make a drop down slide to illustrate that mechanic -->
* Despite this success, regularizers are *vital* for deep learning

<!--

## Regularizers in DNNs

- What are they doing?
* What do they solve?
- How are they used?
-->
<!-- Not just noise -- also completely wrong data -->

---

## Common Techniques

* L1 or L2 penalties
  * These penalize the network for having non-zero weights
* Dropout
  * Portions of network layers are randomly ignored during training
* Stochastic Depth
  * Entire layers of the network are randomly ignored during training
* Changing the learning target
  * For example, predict matrix transform parameters rather than pixel differences

---

## L2 Penalty

<!-- for l2, show example where infinite error can exist between training points. L2 make a smoother surface  -->
<!-- for dropout, show example with incorrect correlation -- different kind of noise -->

<!-- Talk about how it's the learning process rather than the architecture that is vital: convnext vs transformers -->
