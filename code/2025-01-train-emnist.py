#!/usr/bin/python3

# Copyright 2025 Bernhard Firner

import argparse
import functools
import math
import random
import time
import torch
import torchvision
from torchvision.transforms import v2 as transforms

import fleximodel

parser = argparse.ArgumentParser(description='Demonstration of different approaches to classification.')

parser.add_argument(
    '--test',
    type=str,
    required=True,
    choices=['emnist_multiclass'],
    help='Which test to perform')

parser.add_argument(
    '--seed',
    type=int,
    required=False,
    default=0,
    help='Seed for torch.random.manual_seed')

parser.add_argument(
    '--use_amp',
    required=False,
    default=False,
    action="store_true",
    help='Use automatic mixed precision loss')

args = parser.parse_args()

################
# Functions

# Note that torchvision and EMNIST are not playing happilly as of 2024.
# See https://marvinschmitt.com/blog/emnist-manual-loading/index.html
def get_dataset(train=True, split='balanced', transform=None):
    return torchvision.datasets.EMNIST(root="/fastdata/", split=split, train=train, download=False, transform=transform)


def normalizeImages(images, epsilon=1e-05):
    # normalize per channel, so compute over height and width. This handles images with or without a batch dimension.
    v, m = torch.var_mean(images, dim=(images.dim()-2, images.dim()-1), keepdim=True)
    return (images - m) / (v + epsilon)

def updateWithScaler(loss_fn, net, image_input, labels, scaler, optimizer):
    """Update with scaler used in mixed precision training.

    Arguments:
        loss_fn            (function): The loss function used during training.
        net         (torch.nn.Module): The network to train.
        image_input    (torch.tensor): Planar (3D) input to the network.
        labels         (torch.tensor): Desired network output.
        scaler (torch.cuda.amp.GradScaler): Scaler for automatic mixed precision training.
        optimizer       (torch.optim): Optimizer
    """
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out = net(image_input.contiguous())
        loss = loss_fn(out, labels.half())

    scaler.scale(loss.half()).backward()
    scaler.step(optimizer)
    # Important Note: Sometimes the scaler starts off with a value that is too high. This
    # causes the loss to be NaN and the batch loss is not actually propagated. The scaler
    # will reduce the scaling factor, but if all of the batches are skipped then the
    # lr_scheduler should not take a step. More importantly, the batch itself should
    # actually be repeated, otherwise some batches will be skipped.
    # TODO Implement batch repeat by checking scaler.get_scale() before and after the update
    # and repeating if the scale has changed.
    scaler.update()

    return out, loss

def updateWithoutScaler(loss_fn, net, image_input, labels, optimizer):
    """Update without any scaling from mixed precision training.

    Arguments:
        loss_fn          (function): The loss function used during training.
        net       (torch.nn.Module): The network to train.
        image_input  (torch.tensor): Planar (3D) input to the network.
        labels       (torch.tensor): Desired network output.
        optimizer     (torch.optim): Optimizer
    """
    optimizer.zero_grad()
    out = net(image_input.contiguous())
    loss = loss_fn(out, labels)

    loss.backward()
    optimizer.step()

    return out, loss

# Gradient scaler for mixed precision training
if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

def get_example_datum(dataset):
    """Fetch an example image and label for a dataset."""
    probe_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    datum, label = next(probe_dataloader.__iter__())
    return datum, label

def get_dataset_classes(dataset):
    """Iterate through an entire dataset that will be one hot encoded and count the maximum class label."""
    probe_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)
    max_class = -1
    for _, label in probe_dataloader:
        max_class = max(max_class, label.flatten().max().item())
    return max_class + 1

################
# Big model

if args.test == "emnist_multiclass":
    torch.random.manual_seed(args.seed)

    # We need to convert the image fro PIL format into torch tensors
    preprocess = transforms.Compose([
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype()
    ])

    train_dataset = get_dataset(train=True, transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=False)
    # The loss should be derived from nn.LogSoftMax and the nn.NLLLoss functions, or with the equivalent nn.CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Typical neural network with lots of parameters.
    example_img, example_label = get_example_datum(train_dataset)
    max_class = get_dataset_classes(train_dataset)
    print(f"Image size is {example_img[0].size()} and there are {max_class} classes.")
    hyperparams = fleximodel.make_small_hyperparams(in_channels=example_img.size(0), num_outputs=max_class)
    net = fleximodel.FlexiNet(in_dimensions=example_img.size(), out_classes = max_class, hyperparams=hyperparams)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.0)

    net.cuda()
    net.train()

    # Notes:
    # On my computer the times per batch are:
    # CPU with the default dataloader, 4ms/batch
    # GPU with the default dataloader, 2.6ms/batch

    for epoch in range(50):
        print(f"Starting epoch {epoch}")
        begin_time = time.time_ns()
        batch_loss = 0
        for batch_num, dl_tuple in enumerate(train_dataloader):
            images, labels = dl_tuple
            # This isn't strictly necessary, depending upon the dataset, but it generally improves training.
            images = normalizeImages(images.cuda())
            labels = torch.nn.functional.one_hot(labels, num_classes=max_class).float().cuda()

            if scaler is not None:
                out, loss = updateWithScaler(loss_fn, net, images, labels, scaler, optimizer)
            else:
                out, loss = updateWithoutScaler(loss_fn, net, images, labels, optimizer)
            batch_loss += loss
        end_time = time.time_ns()
        duration = end_time - begin_time
        print(f"Average batch loss {batch_loss / (batch_num+1)}")
        print(f"Epoch time {duration/10**9} seconds ({round(duration/10**6/(batch_num+1), ndigits=3)}ms / batch)")

    # The evaluation dataset
    eval_dataset = get_dataset(train=False)
    eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=False)

    net.eval()
    # TODO



