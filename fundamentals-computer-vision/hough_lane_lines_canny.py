import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

IMAGE = "images/exit-ramp.jpg"

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
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
cp_image = np.copy(image)*0 # creating a blank to draw lines on
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(cp_image, (x1,y1), (x2,y2), (255,0,0), 10)

color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

combo = cv2.addWeighted(color_edges, 0.8, cp_image, 1, 0)
plt.imshow(combo, cmap='Greys_r')
plt.show()
