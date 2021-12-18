import numpy as np

def gaussianNoise(image, mean=0, var=0.001):
    ''' 
                 Add Gaussian noise
                 Mean: mean 
                 Var: Variance
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out
