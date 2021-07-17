import numpy as np
import cv2
from utils import calculateHomography

# read the image
src = cv2.imread('images/room.jpg')

height, width = src.shape[:2]

pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# 4 Points corresponding to corners of the television
pts2 = np.float32([ [540, 300], [688, 295], [540, 385], [690, 390] ])

# creating a list of matched points for computing Homography
correspondences = []
for i in range(4):
    x1, y1 = pts1[i]
    x2, y2 = pts2[i]
    correspondences.append([x1, y1, x2, y2])

correspondences = np.matrix(correspondences)
# find homography between two images using ransac
H = calculateHomography(correspondences)
# warp the image using computed homography
img_reg = cv2.warpPerspective(src, H, (width, height))

# create a mask
mask = np.zeros(src.shape, dtype=np.uint8)
roi_corners = np.int32(pts2)
channel_count = src.shape[2]
ignore_mask_color = (255,) * channel_count

# determine the region image must place there
cv2.fillConvexPoly(mask,
                   roi_corners[[0, 1, 3, 2]],
                   ignore_mask_color)

# inverse the mask
mask = cv2.bitwise_not(mask)
# put the background to the image
masked_image = cv2.bitwise_and(src, mask)

#Using Bitwise or to merge the two images
final = cv2.bitwise_or(img_reg, masked_image)

cv2.imshow('img', final)
cv2.waitKey(0)