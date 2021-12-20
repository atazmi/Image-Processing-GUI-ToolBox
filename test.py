import cv2
from matplotlib import pyplot as plt

from backend.histogram import *
from backend.noise import *
from backend.filters import *

img = cv2.imread("images/building.jpg", 0)

'''
histr = calculateHistogram(img)

plt.plot(histr)
plt.show()

equl = equalizeHistogram(img)

histr_equl = calculateHistogram(equl)

plt.plot(histr_equl)
plt.show()
'''

'''
cv2.imshow("orig", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

noisy = gaussianNoise(img, 0, 0.01)

cv2.imshow("noisy", noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

cv2.imshow("orig", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blurredImg = gaussianFilter(img, 15)
cv2.imshow("blurred", blurredImg)
cv2.waitKey(0)
cv2.destroyAllWindows()