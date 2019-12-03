import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

imgs_folder = "./images"
img_name = "test6.jpg"
img_path = os.path.join(imgs_folder, img_name)


def gradient_magnitude(img, thresh, orient='x'):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if 'x' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if 'y' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def color(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary


img = mpimg.imread(img_path)
gradient_binary = gradient_magnitude(img, (20, 100))
color_binary = color(img, (170, 255))
color_and_gradient = np.zeros_like(color_binary)
color_and_gradient[(gradient_binary == 1) | (color_binary == 1)] = 1
plt.imshow(color_and_gradient, cmap="gray")
plt.show()
