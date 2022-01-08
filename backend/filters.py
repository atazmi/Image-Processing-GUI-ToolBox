import copy
import cv2
import numpy as np
from .fourier import shifted_dft, dft_magnitude, inverse_shifted_dft


def gaussianFilter(image, size=5, sigmaX=0, sigmaY=0):
    """
    Takes an image and kernel size and sigma for X and Y directions
    and applies a guissian filter and returns the image.

    Inputs:
        - img: A numpy array of shape (H, W, C) containing the image.
        - size: an integer representing kernel size (default value = 5).
        - sigmaX: an integer representing varience in X direction (default value =0).
        - sigmaY: an integer representing varience in Y direction (default value =0).
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after applying the filter.
    """
    return cv2.GaussianBlur(image, (size, size), sigmaX=sigmaX, sigmaY=sigmaY)


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


def _get_dft_indx(dft_magnitude_value):
    """
    This function to return the index of the dft image it should
    return two pixel indexs
    Args:
        dft_magnitude_value: numpy array
    Outputs:
        indx: tuple of index ((x,y), (x,y)).
    """

    dft_magnitude_value = copy.deepcopy(dft_magnitude_value)
    dft_indexs = ()

    # Get the 3 max pixel values
    while len(dft_indexs) < 3:
        ind = np.unravel_index(dft_magnitude_value.argmax(), dft_magnitude_value.shape)

        # make the max value of pixel to be zero so not be selected again.
        dft_magnitude_value[ind[0], ind[1]] = 0

        # Add the pixel index
        dft_indexs = dft_indexs + (ind,)
    return dft_indexs


def notch_filter(image, offest = 0):
    """
    This function to remove periodic noise by making the dft row and column
     to be zeros
    Args:
        image: numpy array - an image with periodic noise.
    Outputs:
        filtered image: numpy array - an image without periodic noise.
    """
    def _set_dft_to_zero(dft_indx, dft_value, offset = 0):
        """
        This function to make the row and column of dft to be zero
        Args:
        dft_indx: tuple of index.
        dft_value:  numpy array containing the transformed image with the same size but with 2 channels
            (real and complex).
        offest: int - the number of rows|columns to be zeros before and after DFT of the image.

        Outputs:
            dft_value: numpy array containing the transformed image
        """
        dft_indx = list(dft_indx)
        dft_indx1 = []
        if offset:

            for i in range(1, offset + 1):
                for dft_ind in dft_indx:
                    dft_indx1.append((dft_ind[0] + i, dft_ind[1]))
                    dft_indx1.append((dft_ind[0] - i, dft_ind[1]))

                    dft_indx1.append((dft_ind[0], dft_ind[1] + i))
                    dft_indx1.append((dft_ind[0], dft_ind[1] - i))

        dft_indx.extend(dft_indx1)
        for dft_ind in dft_indx:
            dft_value[dft_ind[0], :] = 0
            dft_value[:, dft_ind[1]] = 0
        return dft_value

    # Get the DFT of the image
    dft_value = shifted_dft(image)

    # Get the magnitude of the DFT
    dft_magnitude_value = dft_magnitude(dft_value)

    # Get the index of the DFT
    dft_indx = _get_dft_indx(dft_magnitude_value)[1:]

    # Set the dft to zero
    filtered_dft = _set_dft_to_zero(dft_indx, dft_value, offest)

    # get the image back
    img_back = inverse_shifted_dft(filtered_dft)

    return img_back


def band_filter(image, offset = 3):
    """
    This function to remove periodic noise by drawing a circle passing through
    the DFT of the image and then making this circle equal to zero.
    Args:
        image: numpy array - an image with periodic noise.
    Outputs:
        filtered image: numpy array - an image without periodic noise.
    """

    def _draw_band(image_shape, points, offest=3):
        """
        This function to draw the band on image with all ones except the band to be zeros
        Args:
            image_shape: tupel TFD points, each with width and height.
            points: Tuple - DFT points.
            offest: int - the thikness of the band.

        """
        center = points[0]
        blank_image = np.ones(image_shape)
        for point in points[1:]:
            radius = np.linalg.norm(np.array(point) - center)
            blank_image = cv2.circle(blank_image, (points[0][1], points[0][0]), int(round(radius)), (0, 0, 0), offest)
        return blank_image

    # Get the DFT of the image
    dft_value = shifted_dft(image)

    # Get the magnitude of the DFT
    dft_magnitude_value = dft_magnitude(dft_value)

    # Get the index of the DFT
    dft_indx = _get_dft_indx(dft_magnitude_value)

    # draw the DFT band
    blank_image_with_band = _draw_band(image.shape, dft_indx, offset)

    # set zeros to the DFT band
    dft_value *= blank_image_with_band

    # get the image back
    img_back = inverse_shifted_dft(dft_value)

    return img_back

def maskFilter(image_fourier, point_1, point_2, filter_size = 50):
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
    filter_size = int(filter_size)
    mask = np.ones_like(image_fourier, dtype=np.uint8)
    if(len(image_fourier.shape)==2):
        mask[point_1[0]-filter_size:point_1[0]+filter_size, point_1[1]-filter_size:point_1[1]+filter_size] = 0
        mask[point_2[0]-filter_size:point_2[0]+filter_size, point_2[1]-filter_size:point_2[1]+filter_size] = 0
    else:
        mask[point_1[0]-filter_size:point_1[0]+filter_size, point_1[1]-filter_size:point_1[1]+filter_size, :] = 0
        mask[point_2[0]-filter_size:point_2[0]+filter_size, point_2[1]-filter_size:point_2[1]+filter_size, :] = 0
    # cv2.imshow("Mask", 255*mask)
    # cv2.waitKey(0)
    ouput_image = mask * image_fourier
    # cv2.imshow("after filter", np.array(ouput_image, dtype=np.uint8))
    # cv2.waitKey(0)
    return ouput_image