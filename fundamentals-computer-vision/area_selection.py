import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread("test.jpg")
print("This image is ", type(image),
      "with dimensions: ", image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

area_select = np.copy(image)

# (0, 0) top left
left_bottom = [200, 520]
right_bottom = [800, 520]
apex = [500, 320]

# y=Ax+B
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
             (YY > (XX*fit_right[0]+fit_right[1])) & \
             (YY < (XX*fit_bottom[0]+fit_bottom[1]))

area_select[thresholds] = [255, 0, 0]

plt.imshow(area_select)
plt.show()
plt.savefig("output.jpg")
