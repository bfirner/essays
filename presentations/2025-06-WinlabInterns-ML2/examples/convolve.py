#!/usr/bin/python3

# This code runs a provided filter over an image and saves the output.

import argparse

import numpy
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convolution demo.")
    parser.add_argument(
        '--image_path',
        required=True,
        type=str,
        help='Path to an image file')
    parser.add_argument(
        '--outpath',
        type=str,
        default="filtered.png",
        help='Path to save the filtered image')
    parser.add_argument(
        '--filter',
        required=True,
        nargs=9,
        type=float,
        help='A 3x3 filter with 9 values')
    args = parser.parse_args()

    kernel = numpy.array(args.filter).reshape((3,3))

    image = cv2.imread(args.image_path)
    if image is None:
        print("Failed to load image.")
        exit()

    # Filter and keep the output depth the same
    filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imwrite(args.outpath, filtered)
