import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

images_folder = "./images"
image_name = "signs_vehicles_xygrad.png"
image_path = os.path.join(images_folder, image_name)

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_factor = np.max(gradmag)/255
    gradmag = (gradmag/scaled_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    # 6) Return this mask as your binary_output image
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag < mag_thresh[1])] = 1

    return binary_output

if __name__ == "__main__":
    img = mpimg.imread(image_path)
    output = mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    plt.imshow(output, cmap="gray")
    plt.show()
