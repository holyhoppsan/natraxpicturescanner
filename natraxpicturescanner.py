from transform import four_point_transform

import numpy as np
import argparse
import cv2

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", help="path to the image file")
argument_parser.add_argument(
    "-c", "--coords", help="comma separated lists of source points"
)

arguments = vars(argument_parser.parse_args())

image = cv2.imread(arguments["image"])

points = np.array(eval(arguments["coords"]), dtype="float32")

warped = four_point_transform(image, points)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
