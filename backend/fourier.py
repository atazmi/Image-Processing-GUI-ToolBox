import numpy as np

def shifted_dft(img):
    """
    Transform image from spatial domain to frequency domain by preforming forward Discrete Fourier transform and then
    shift the zero-frequency component to the center of the spectrum.

    Inputs:
        - img: A numpy array of any shape containing the image.

    Returns:
        - shifted_dft: a numpy array containing the transformed image with the same size but with 2 channels
        (real and complex).
    """

    img_dft = np.fft.fft2(img)
    shifted_dft = np.fft.fftshift(img_dft)

    return shifted_dft


def dft_magnitude(shifted_dft):
    """
    Compute the magnitude spectrum of an image in the frequency domain for analysis and visualization purposes.

    Inputs:
        - shifted_dft: A numpy array of shape (H, W, 2) containing the image in frequency domain with the
        zero-frequency component shifted to the center of the spectrum.

    Returns:
        - magnitude_spectrum: a numpy array of shape (H, W) containing intensity values.
    """

    magnitude_spectrum = 20* np.log(np.abs(shifted_dft)+1)
    return magnitude_spectrum


def inverse_shifted_dft(shifted_dft):
    """
    Calculates the inverse Discrete Fourier Transform of an image after shifting the zero frequency component (DC component) to be at top left corner.

    Inputs:
        - shifted_dft: A numpy array of shape (H, W, 2) containing the image in frequency domain with the
        zero-frequency component shifted to the center of the spectrum.

    Returns:
        - img: a numpy array of shape (H, W) containing the original image.
    """

    img_dft = np.fft.ifftshift(shifted_dft)
    img = np.fft.ifft2(img_dft)
    img = np.abs(img)

    return img
