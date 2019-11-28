import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

IMAGE = "exit-ramp.jpg"

image = mpimg.imread(IMAGE)

# grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap='gray')

# configure Gaussian smoothing
kernel_size = 3

blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# detect edges with Canny
low_threshold = 90
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='Greys_r')
plt.show()
