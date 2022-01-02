import cv2


def gaussianFilter(image, size=5, sigmaX=0, sigmaY=0):
    return cv2.GaussianBlur(image, (size, size), sigmaX=0, sigmaY=0)


def sobel_filter(image, ksize=3, dx=1, dy=0):
    return cv2.Sobel(src=image, ddepth=cv2.CV_8U, dx=dx, dy=dy, ksize=ksize)


def laplacian_filter(image, ksize=3):
    return cv2.Laplacian(image, ddepth=cv2.CV_8U, ksize=ksize)


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
