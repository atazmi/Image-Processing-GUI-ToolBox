import cv2

def calculateHistogram(image):
    """
    This function uses OpenCV library.
    Takes an image and checks if colored or gray, if colored convert to grayscale image then
    calculates its histogram and returns the histogram. 

    Inputs:
        - img: A numpy array.
    Returns:
        - image: a numpy array.
    """
    # if(len(image.shape)==3):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([image],[0],None,[256],[0,256])

def equalizeHistogram(image):
    """
    This function uses OpenCV library.
    Takes an image and applies Histogram Equalization to better distribute intensities 
    accross the spectrum and returns the modified image. 

    Inputs:
        - img: A numpy array.
    Returns:
        - image: a numpy array.
    """
    return cv2.equalizeHist(image)
