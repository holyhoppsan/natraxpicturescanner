from transform import four_point_transform
from skimage.filters import threshold_local

import numpy as np
import argparse
import cv2
import helpers

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    "-i", "--image", required=True, help="path to the image file"
)
argument_parser.add_argument(
    "-c", "--coords", help="comma separated lists of source points"
)

arguments = vars(argument_parser.parse_args())

if arguments["coords"] is not None:

    image = cv2.imread(arguments["image"])

    points = np.array(eval(arguments["coords"]), dtype="float32")

    warped = four_point_transform(image, points)

    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # performing edge detection
    print("Performing edge detevtion")

    image = cv2.imread(arguments["image"])
    ratio = image.shape[0] / 500.0
    original_image = image.copy()

    image = helpers.resize(image, output_height=500)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.GaussianBlur(grayscale_image, (5, 5), sigmaX=0, sigmaY=0)
    edged_image = cv2.Canny(grayscale_image, 100, 200)

    print("Step1: Edge detection")
    cv2.imshow("Image", image)
    cv2.imshow("Grayshade", grayscale_image)
    cv2.imshow("Edged", edged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
