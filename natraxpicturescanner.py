from flask import Flask, render_template
from transform import four_point_transform
from skimage.filters import threshold_local

import numpy as np
import argparse
import cv2
import helpers

# Initialization
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-i", "--image", required=True, help="path to the image file"
    )
    argument_parser.add_argument(
        "-c", "--coords", help="comma separated lists of source points"
    )

    arguments = vars(argument_parser.parse_args())

    return arguments


def read_image_from_arguments(arguments):
    return cv2.imread(arguments["image"])


def warp_image_with_manual_input(image):

    points = np.array(eval(arguments["coords"]), dtype="float32")

    warped = four_point_transform(image, points)

    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perform_edge_detection(image):
    ratio = image.shape[0] / 500.0
    original_image = image.copy()

    image = helpers.resize(image, output_height=500)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.GaussianBlur(grayscale_image, (5, 5), sigmaX=0, sigmaY=0)
    edged_image = helpers.auto_canny(
        grayscale_image
    )  # cv2.Canny(grayscale_image, 50, 200)

    print("Step1: Edge detection")
    cv2.imshow("Image", image)
    cv2.imshow("Grayshade", grayscale_image)
    cv2.imshow("Edged", edged_image)
    # cv2.waitKey(0)
    return edged_image


def find_contours(edged_image):
    print("Step2: Find contours of paper")
    contours = cv2.findContours(
        edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = helpers.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenContour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            screenContour = approx
            break

    return screenContour


def show_image_with_contours(image, contours):
    print("Show results of contours")
    cv2.drawContours(image, [contours], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    arguments = parse_args()

    input_image = read_image_from_arguments(arguments)

    if arguments["coords"] is not None:
        warp_image_with_manual_input(input_image)
    else:
        edged_image = perform_edge_detection(input_image)
        contours = find_contours(edged_image)
        show_image_with_contours(input_image, contours)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
