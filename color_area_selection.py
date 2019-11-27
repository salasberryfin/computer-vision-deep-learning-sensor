import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread("test.jpg")
print("This image is ", type(image),
      "with dimensions: ", image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)
line_image = np.copy(image)

# detect clear colors
red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold ]

# mark all pixels that are darker than the configured RGB threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
              | (image[:,:,1] < rgb_threshold[1]) \
              | (image[:,:,2] < rgb_threshold[2])

# select are
left_bottom = [150, 520]
right_bottom = [800, 520]
apex = [480, 320]

# y=Ax+B
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
area_thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
             (YY > (XX*fit_right[0]+fit_right[1])) & \
             (YY < (XX*fit_bottom[0]+fit_bottom[1]))

# paint all those pixels black
color_select[color_thresholds & ~area_thresholds] = [0, 0, 0]

# merge both selections: image colored and in region
line_image[~color_thresholds & area_thresholds] = [255, 0, 0]

plt.imshow(line_image)
plt.savefig("line-image.jpg")
plt.imshow(color_select)
plt.savefig("color-select.jpg")
