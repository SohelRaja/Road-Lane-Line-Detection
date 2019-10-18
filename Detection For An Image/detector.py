# Import Libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Only for required region
def require_part(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = (255, 255, 255)
    # for filling the polygon
    cv.fillPoly(mask, vertices, match_mask_color)

    mask_img = cv.bitwise_and(image, mask)
    return mask_img

# For Draw The Line
def draw_lines(image, lines):
    image = np.copy(image)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=15)
    image = cv.addWeighted(image, 0.8, line_image, 1, 0.0)
    return image

# Loading the image
img = cv.imread('../images/road1.jpg')

# Converting the image to rgb
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Printing image info
print(img.shape)
height = img.shape[0]
width = img.shape[1]

# Defining the required region
require_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

# For finding the edges
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny_image = cv.Canny(gray_image, 60, 100)

# Mask Image
mask_image = require_part(canny_image, np.array([require_vertices], dtype=np.int32))

# Draw Lines
lines = cv.HoughLinesP(mask_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)

image_with_line = draw_lines(img, lines)
# Showing the image using matplotlib
plt.imshow(image_with_line)
plt.show()
