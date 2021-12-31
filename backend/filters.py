import cv2
import numpy as np

def gaussianFilter(image, size=5, sigmaX=0, sigmaY=0):
    """
    Takes an image and the gaussian filter paramters and apply gaussian filter to the image and returns the filtered image.

    Inputs:
        - image: A numpy array of shape (H, W, C) containing the image.
        - size: an integer representing kernel size (default value = 5).
        - sigmaX: an integer representing variance in x direction (default value = 0).
        - sigmaY: an integer representing variance in y direction (default value = 0).
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after applying the filter.
    """
    return cv2.GaussianBlur(image, (size, size), sigmaX=sigmaX, sigmaY=sigmaY)


def sobel_filter(image, ksize=3, dx=1, dy=0):
    return cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=ksize)


def laplacian_filter(image, ksize=3):
    return cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=ksize)


def averaging_filter(img, kernel_height=5, kernel_width=5):
    """
    Takes an image and kernel height and width (default value for kernel size = 5 * 5) and apply an averaging filter.

    Inputs:
        - img: A numpy array of shape (H, W, C) containing the image.
        - kernel_height: an integer representing kernel height (default value = 5).
        - kernel_width: an integer representing kernel width (default value = 5).
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after applying the filter.
    """
    image = cv2.blur(img, (kernel_height, kernel_width))
    return image


def median_filter(img, kernel_size=5):
    """
    Takes an image and kernel size (default value for kernel size = 5 * 5) and apply a median blur filter.

    Inputs:
        - img: A numpy array of shape (H, W, C) containing the image.
        - kernel_size: an odd integer representing kernel size (default value = 5).
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after applying the filter.
    """
    image = cv2.medianBlur(img, kernel_size)
    return image


def maskFilter(image, point_1, point_2):
    """
    Takes an image and two points in the image to suppress as a maskFilter.

    Inputs:
        - image: A numpy array of shape (H, W) containing the image.
        - point_1: array of two values for the first point (x,y) to be supressed.
        - point_2: array of two values for the first point (x,y) to be supressed.
    Returns:
        - mask: a numpy array of shape (H, W) containing the mask after supressing the
                the two points.
    """
    mask = np.ones_like(image)
    mask[point_1[0], point_1[1]] = mask[point_2[0], point_2[1]] = 0
