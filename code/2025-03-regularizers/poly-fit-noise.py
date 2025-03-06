#!/usr/bin/python3

import numpy

def sample_curve(offset, spread, magnitude, x):
    """Produce a curve that looks like the overfitting example in "The Little Book of Deep Learning" by François Fleuret."""
    return magnitude * 2**(-(x - offset)**2/spread)


# The x and y points along a curve
x_samples = [0.05 * x for x in range(21)]
y_samples = [sample_curve(0.5, 0.1, 1, x) for x in x_samples]
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

