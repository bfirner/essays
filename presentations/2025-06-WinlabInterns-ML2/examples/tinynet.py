"""
Tiny network for transfer learning
"""

import torch


# Input is a 56x56 images

tiny = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=0, stride=1),  # 32x52x52
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x26x26
    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0, stride=1), # 32x24x24
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x12x12
    torch.nn.Flatten(),                                                                   # 4608 features
    torch.nn.Linear(in_features=32*12*12, out_features=25))                               # 25 classes

