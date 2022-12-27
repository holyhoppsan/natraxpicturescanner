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
