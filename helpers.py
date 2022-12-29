import numpy as np
import cv2


def resize(image, output_width=None, output_height=None, inter=cv2.INTER_AREA):
    dimensions = None
    (input_height, input_width) = image.shape[:2]

    if output_width is None and output_height is None:
        return image

    if output_width is None:
        ratio = output_height / float(input_height)
        dimensions = (int(input_width * ratio), output_height)

    else:
        ratio = output_width / float(input_width)
        dimensions = (output_width, int(input_height * ratio))

    resized_image = cv2.resize(image, dimensions, interpolation=inter)

    return resized_image


def grab_contours(contours):
    if len(contours) == 2:
        contours = contours[0]

    elif len(contours) == 3:
        contours = contours[1]

    else:
        raise Exception(
            (
                "Contours tuple must have length 2 or 3, "
                "otherwise OpenCV changed their cv2.findContours return "
                "signature yet again. Refer to OpenCV's documentation "
                "in that case"
            )
        )

    return contours


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
