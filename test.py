import cv2
from matplotlib import pyplot as plt

from backend.fourier import dft_magnitude, shifted_dft
from backend.histogram import *
from backend.noise import *
from backend.filters import *

img = cv2.imread("Images/building.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("grey", img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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

# cv2.imshow("orig", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# blurredImg = gaussianFilter(img, 15)
# cv2.imshow("blurred", blurredImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# soble_edge_img = sobel_filter(img, 11, 1, 1)
# plt.imshow(soble_edge_img)
# plt.show()
# cv2.imshow("soble_edge_img", soble_edge_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# laplace_edge_img = laplacian_filter(img, ksize=7)
# cv2.imshow("laplace_edge_img", laplace_edge_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#

# salt_and_pepper_img = add_salt_and_pepper_noise(img, 0.30, 0.8)
# plt.subplot(1, 3, 1)
# plt.imshow(salt_and_pepper_img)
# plt.subplot(1, 3, 2)
# plt.imshow(median_filter(salt_and_pepper_img,  5))
# plt.subplot(1, 3, 3)
# plt.imshow(averaging_filter(salt_and_pepper_img,  5, 5))
# plt.show()


# plt.imshow(add_periodic_noise(img_gray), cmap = 'gray')
# plt.show()
# noisy = notch_filter(add_periodic_noise(img_gray), 0)
# plt.imshow(noisy, cmap = 'gray')
# plt.show()
#

#
# filtered_image = band_filter(img_with_periodic_noise)
# cv2.imwrite('image_filtered_with_band_filter.jpg', filtered_image)
# image_filtered_with_notch = cv2.imread("image_filtered_with_band_filter.jpg")
# cv2.imshow("image_filtered_with_band_filter", image_filtered_with_notch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img_with_periodic_noise = add_periodic_noise(img_gray, 200, 50, 70, 80)

plt.imshow(img_with_periodic_noise, cmap = 'gray')
plt.show()
# cv2.imshow("img_with_periodic_noise", img_with_periodic_noise)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


filtered_image = notch_filter(img_with_periodic_noise, 10)
plt.imshow(filtered_image, cmap = 'gray')
plt.show()
# cv2.imwrite('image_filtered_with_notch.jpg', filtered_image)
# cv2.imshow("image_filtered_with_notch", image_filtered_with_notch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
