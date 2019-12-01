import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_folder = "./images"
image_name = "test_image2.png"

nx = 8
ny = 6


def find_corners(gray, n):
    objpoints = []  # 3D real points
    imgpoints = []  # 2D image points
    objp = np.zeros((n[0]*n[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:n[0], 0:n[1]].T.reshape(-1, 2)
    ret, corners = cv2.findChessboardCorners(gray, (n[0], n[1]), None)
    if ret is True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        return imgpoints, objpoints

    return None


def calibrate_undistort(imgpoints, objpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)
    dst = cv2.undistort(image,
                        mtx,
                        dist,
                        None,
                        mtx)

    return dst
    

def draw_undist_corners(img, n):
    ret, corners = cv2.findChessboardCorners(img, n, None)
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    if ret is True:
        cv2.drawChessboardCorners(img, n, corners, ret)

        return img, src


def transform_perspective(img, src):
    offset = 100
    img_size = (img.shape[1], img.shape[0])
    dest = np.float32([[offset, offset], [img_size[0]-offset, offset],
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped


if __name__ == '__main__':
    image_path = os.path.join(images_folder, image_name)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgpoints, objpoints = find_corners(gray, (nx, ny))
    dst = calibrate_undistort(imgpoints, objpoints)
    undist_corners, src = draw_undist_corners(dst, (nx, ny))
    warped = transform_perspective(undist_corners, src)
    plt.imshow(warped)
    plt.show()

