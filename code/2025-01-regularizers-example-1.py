#!/usr/bin/python3

# This is a more extensive example of overfitting, continuing from 24-12-regularizers-example.py
# Simple, linear functions don't have an overfitting problem. A high- or low- pass filter, with complex dynamic behavior before a rapid change in response, is a good example of a function that a DNN could have trouble approximating without sufficient data.
# More interesting is a visual classification problem.
# Draw arrows on an image and ask the DNN to predict the angle (as on the face of a clock).
# If there is a dead space in the training data, it'll flop.

import functools
import random
import torch

