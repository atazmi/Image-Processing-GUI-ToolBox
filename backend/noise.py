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



def add_periodic_noise(image, A=100, u0=90, v0=50):
    '''
            A: amplitude - int.
            u0: angle 
            v0: angle

    '''
    shape = image.shape
    noise = np.zeros(shape, dtype='float64')
    x, y = np.meshgrid(range(0, shape[1]), range(0, shape[0]))

    noise += A*np.sin(x*u0 + y*v0)

    return image+noise

