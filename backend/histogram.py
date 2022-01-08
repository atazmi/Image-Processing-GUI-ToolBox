import cv2
import numpy as np


def calculateHistogram(image):
    """
    Takes an image and calculates its histogram.

    Inputs:
        - image: A numpy array of shape (H, W, C) containing the image.
    Returns:
        - histogram: a numpy array of shape (H, W).
    """
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    return cv2.calcHist([image],[0],None,[256],[0,256])

def equalizeHistogram(image):
    """
    Takes an image and applies histogram equalization.

    Inputs:
        - image: A numpy array of shape (H, W, C) containing the image.
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after applying 
                 histogram equalization.
    """
    return cv2.equalizeHist(np.uint8(image))
