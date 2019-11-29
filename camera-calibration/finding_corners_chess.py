import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_folder = "./images"
image_name = "calibration_test.png"
image_path = os.path.join(images_folder, image_name)

nx = 8
ny = 6

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

retval, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if retval == True:
    cv2.drawChessboardCorners(image, (nx, ny), corners, retval)
    plt.imshow(image)
    plt.show()

