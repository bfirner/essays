
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
    args = parser.parse_args()

    # Four basic filters
    kernels = numpy.array(
            [[[0,  1, 0],   # Lower left diagonal
              [1,  0, -1],
              [0, -1, 0]],
             [[0,  1, 0],   # Upper left diagonal
              [-1, 0, 1],
              [0, -1, 0]],
             [[1, 0, -1],   # Horizontal
              [1, 0, -1],
              [1, 0, -1]],
             [[ 1,  1,  1], # Vertical
              [ 0,  0,  0],
              [-1, -1, -1]]])

    image = cv2.imread(args.image_path)
    if image is None:
        print("Failed to load image.")
        exit()

    # Normalize the range from 0-255 to 0-1
    image = image / 255.0

    # Run through the same filters 2 times
    for step in range(1):
        outputs = []
        for i in range(4):
            # Filter and keep the output depth the same
            filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernels[i])
            # Set any negative values to 0
            filtered[filtered < 0] = 0
            # Set anything larger than 1 to 1
            filtered[filtered > 1] = 1
            outputs.append(filtered)

    for step in range(3):
        fuzzy_bias = -1
        fuzzy_kernel = numpy.array(
            [[0.4, 0, 0.4],
             [0, 0.4, 0],
             [0.4, 0, 0.4]])
        fuzzy_outputs = [cv2.filter2D(src=output, ddepth=-1, kernel=fuzzy_kernel) for output in outputs]
        fuzzy_sum = numpy.sum(fuzzy_outputs, axis=0) + fuzzy_bias
        fuzzy_sum[fuzzy_sum < 0] = 0
        fuzzy_sum[fuzzy_sum > 1] = 1
        outputs = [fuzzy_sum]

    for step in range(3):
        fuzzy_bias = -1
        fuzzy_kernel = numpy.array(
            [[0, 0.5, 0],
             [0.5, 0.4, 0.5],
             [0, 0.5, 0]])
        fuzzy_outputs = [cv2.filter2D(src=output, ddepth=-1, kernel=fuzzy_kernel) for output in outputs]
        fuzzy_sum = numpy.sum(fuzzy_outputs, axis=0) + fuzzy_bias
        fuzzy_sum[fuzzy_sum < 0] = 0
        fuzzy_sum[fuzzy_sum > 1] = 1
        outputs = [fuzzy_sum]

    cv2.imwrite(args.outpath, fuzzy_sum*255)
