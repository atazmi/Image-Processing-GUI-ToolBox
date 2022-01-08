import numpy as np


def gaussianNoise(image, mean=0, var=0.001):
    """
    Takes an image and gussian noise to it with the defined mean and variance.

    Inputs:
        - image: A numpy array of any shape containing the image.
        - mean: The mean of the added noise
        - var: The variance of the added noise

    Returns:
        - Image: a numpy array containing the image after adding the gaussian noise.
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def add_periodic_noise(image, A=100, u0=90, v0=50, freq = 100):
    '''
            A: amplitude - int.
            u0: angle 
            v0: angle

    '''
    shape = image.shape
    noise = np.zeros((shape[0], shape[1]), dtype='float32')
    x, y = np.meshgrid(range(0, shape[1]), range(0, shape[0]))

    noise += A * np.sin((x * u0 + y * v0) / freq)

    if len(image.shape) > 2:
        noise = noise.reshape(shape[0], shape[1], 1)

    return image + noise


def add_salt_and_pepper_noise(img, noise_intensity=0.50, salt_intensity=0.5):
    """
    Takes an image and adds salt and pepper noise to it. Noise is added to a certain number of pixels chosen randomly
    based the value of noise_intensity (default value for noise_intensity = 0.5) which means that half of the pixels will be noisy and
    the rest won't be affected. Pixels are chosen randomly then a certain number of pixels will be turned into black
    pixels based on salt_intensity value (default value for salt_intensity = 0.5) and the rest will be turned into white.

    Inputs:
        - img: A numpy array of shape (H, W, C) containing the image.
        - noise_intensity: a float between 0 and 1 specifying the ratio of noise added to the image.
        - salt_intensity: a float between 0 and 1 specifying the ratio of salt noise. Ratio of pepper noise = (1 - salt_intensity)
    Returns:
        - image: a numpy array of shape (H, W, C) containing the image after adding the noise.
    """
    black = 0
    white = 255
    image = img.copy()
    if len(image.shape) == 3:
        black = [0] * img.shape[2]
        white = [255] * img.shape[2]

    h, w = image.shape[0:2]

    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    coordinates = np.dstack((xx, yy)).reshape(-1, 2)

    sample_size = int(noise_intensity * h * w)
    sample = np.random.choice(np.arange(h * w), sample_size, replace=False)

    salt_sample_size = int(salt_intensity * sample_size)
    salt_sample = np.random.choice(sample, salt_sample_size, replace=False)
    pepper_sample = np.setdiff1d(sample, salt_sample)

    image[coordinates[salt_sample][:, 0], coordinates[salt_sample][:, 1]] = white
    image[coordinates[pepper_sample][:, 0], coordinates[pepper_sample][:, 1]] = black

    return image
