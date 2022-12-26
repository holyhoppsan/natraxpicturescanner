import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum)]
    rect[2] = pts[np.argmax(sum)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (top_left, top_right, bottom_left, bottom_right) = rect

    width_a = np.sqrt(
        ((bottom_right[0] - bottom_left[0]) ** 2)
        + ((bottom_right[1] - bottom_left[1]) ** 2)
    )

    width_b = np.sqrt(
        ((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2)
    )

    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(
        ((top_right[0] - bottom_right[0]) ** 2)
        + ((top_right[1] - bottom_right[1]) ** 2)
    )

    height_b = np.sqrt(
        ((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2)
    )

    max_height = max(int(height_a), int(height_b))

    destination_rect = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    perspective_transformation_matrix = cv2.getPerspectiveTransform(
        rect, destination_rect
    )

    warped_image = cv2.warpPerspective(
        image, perspective_transformation_matrix, (max_width, max_height)
    )

    return warped_image
