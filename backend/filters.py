import cv2

def gaussianFilter(image, size=5, sigmaX=0, sigmaY=0):
    return cv2.GaussianBlur(image,(size,size), sigmaX=0, sigmaY=0)


def sobel_filter(image, ksize=3, dx=1, dy=0):
    return cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=ksize)

def laplacian_filter(image, ksize=3):
    return cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=ksize)