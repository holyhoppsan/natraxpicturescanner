import numpy as np
import cv2 

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")

    sum = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(sum)]
    rect[2] = pts[np.argmax(sum)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


