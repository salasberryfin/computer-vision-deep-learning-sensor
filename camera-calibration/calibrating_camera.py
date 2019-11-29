import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_folder = "./images/calibration_wide"
images_exp = "GOPR00*.jpg"
images_path = os.path.join(images_folder, images_exp)

images = glob.glob(images_path)

objpoints = []  # 3D real points
imgpoints = []  # 2D image points

# generate object points
# chessboard 8*6, with 3 coordinates x, y, z
objp = np.zeros((8*6, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

for image in images:
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    
    if ret is True:
        # cv2.drawChessboardCorners(image, (8, 6), corners, ret)
        imgpoints.append(corners)
        objpoints.append(objp)
        # Calibrate Cameta
        # mtx = camera matrix
        # dist = distortion coefficient
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                           imgpoints,
                                                           gray.shape[::-1],
                                                           None,
                                                           None)
        # Correct image distortion
        dst = cv2.undistort(image, mtx, dist, None, mtx)

        # plt.imshow(dst)
        # plt.show()
