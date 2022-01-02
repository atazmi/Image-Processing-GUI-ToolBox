import cv2
import numpy as np


def calculateHistogram(image):
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    return cv2.calcHist([image],[0],None,[256],[0,256])

def equalizeHistogram(image):
    return cv2.equalizeHist(np.uint8(image))
