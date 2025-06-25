"""
Save LeNet 5 in onnx format for visualization.
"""

import torch


# Input is a 24x24 single channel image

lenet = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, stride=1),  # 28x28x4, 104 parameters
    torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),                              # 14x14x4
    torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5, padding=0, stride=1), # 9x9x12, should be 20x(5x5kernels) + 12 bias = 512 parameters with selected connections
    torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),                              # 4x4x12
    torch.nn.Flatten(),                                                                  # 192 features
    torch.nn.Linear(in_features=4*4*12, out_features=10))                                # 1920 weights + 10 bias parameters

probe = torch.randn(1, 1, 24, 24)

lenet.forward(probe)

torch.onnx.export(lenet, probe, "lenet5.onnx", input_names=["patch"], output_names=["prediction"])
