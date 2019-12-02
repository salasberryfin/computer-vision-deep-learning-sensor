import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

images_folder = "./images"
image_name = "signs_vehicles_xygrad.png"
image_path = os.path.join(images_folder, image_name)


# 1) Convert to grayscale
# 2) Take the derivative in x or y given orient = 'x' or 'y'
# 3) Take the absolute value of the derivative or gradient
# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
# 5) Create a mask of 1's where the scaled gradient magnitude 
        # is > thresh_min and < thresh_max
# 6) Return this mask as your binary_output image
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if 'x' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif 'y' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
    plt.imshow(binary_output, cmap='gray')
    plt.show()


if __name__ == "__main__":
    img = cv2.imread(image_path)
    abs_sobel_thresh(img, thresh_min=20, thresh_max=100)
