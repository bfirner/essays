#!/usr/bin/python3

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
