import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

imgs_folder = "./images"
img_name = "test6.jpg"
img_path = os.path.join(imgs_folder, img_name)


def gray_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return gray


def hls_img(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    return hls


def hls_thresh(img, thresh):
    # H detects lines as black (not very well)
    H = img[:, :, 0]
    L = img[:, :, 1]
    # S detects lines fairly well
    S = img[:, :, 2]
    binaryS = np.zeros_like(S)
    binaryS[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binaryS


img = mpimg.imread(img_path)
gray = gray_img(img)
hls = hls_img(img)
binary_hls = hls_thresh(hls, (90, 255))

plt.imshow(binary_hls, cmap="gray")
plt.show()
