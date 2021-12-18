import cv2

def gaussianFilter(image, size=5, sigmaX=0, sigmaY=0):
    return cv2.GaussianBlur(image,(size,size), sigmaX=0, sigmaY=0)
