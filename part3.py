import numpy as np
import cv2
from utils import *

# path to the video & image we want to augment
VIDEO_PATH = 'images/Video1.avi'
IMAGE_PATH = 'images/building2.jpg'
# open video file
vid = cv2.VideoCapture(VIDEO_PATH)

# read the image we want to augment that
building = cv2.imread(IMAGE_PATH, 0)

# read first frame of the video
_, frame1 = vid.read()

# params for corner detection
feature_params = dict(maxCorners=250,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(17, 17),
                 maxLevel=7,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# convert frame from BGR to Gray level
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# take width and height of video frame
height, width = frame1_gray.shape

# resize our image to became the same size of the video frame
building = cv2.resize(building, (width, height))

# select 4 points of region we want to augment image on that area
# order of choosing points: 1)TOP LEFT  2)TOP RIGHT  3)BOTTOM RIGHT  4)BOTTOM LEFT
pts = np.array(select_region(VIDEO_PATH))
cv2.destroyAllWindows()
# 4 corner points of the building image. we use this points to warp the image with respect to selected points
pts1 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
# compute transform between this 2 sets point to better placement the building image in first frame
transform = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts)) # Argument pass must be float
# warp the image
building = cv2.warpPerspective(building, transform, (width, height))

# create a mask to determine where find features for first frame
mask_frame = np.zeros_like(frame1_gray)
mask_frame[pts[1][1]: pts[2][1], pts[0][0]: pts[1][0]] = 255
pts = np.float32(pts.reshape((-1, 1, 2)))


# Take first frame and find corners in it
old_points = cv2.goodFeaturesToTrack(frame1_gray, mask=mask_frame, **feature_params)
temp = old_points

old_frame_gray = frame1_gray

# number of frames of the video
length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
for _ in range(length-1):
    # read the next frame and convert it to gray level
    _, new_frame = vid.read()
    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    # make a copy of frame for later use
    img_aug = new_frame_gray.copy()

    # calculate optical flow
    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, new_frame_gray, old_points, None, **lk_params)

    # using points that we found (st==1)
    good_new = new_points[st == 1]
    good_old = temp[st == 1]

    #############################
    # temporary: for showing tracking points
    # IMAGE = new_frame_gray.copy()
    # for x, y in good_new:
    #     IMAGE = cv2.circle(IMAGE, (np.int32(x), np.int32(y)), 2, (0, 0, 255), 3)
    #     cv2.imshow('tracking features', IMAGE)
    #############################

    # creating a list of matched points between 2 frame for computing Homography
    correspondenceList = []
    for i in range(len(good_old)):
        x1, y1 = good_old[i]
        x2, y2 = good_new[i]
        correspondenceList.append([x1, y1, x2, y2])


    corrs = np.matrix(correspondenceList)
    # find Homography between 2 frames
    finalH, inliers = ransac(corrs, 1, 10, verbose=False)

    # new frame will be old frame for next frame!
    old_frame_gray = new_frame_gray
    old_points = new_points

    # warp building with calculated homography
    img_warp = cv2.warpPerspective(building, finalH, (new_frame_gray.shape[1], new_frame_gray.shape[0]))

    # make mask for determine witch part will be background and witch part will be building image
    mask = np.zeros((new_frame_gray.shape[0], new_frame_gray.shape[1]), np.uint8)
    # warp selected points to find their new coordinates
    new_pts = cv2.perspectiveTransform(pts, finalH)
    # make white the region that building image must place there
    cv2.fillPoly(mask, [np.int32(new_pts)], (255, 255, 255))
    # but we want inverse of it!
    mask_inv = cv2.bitwise_not(mask)
    # with this we have frame (or background) that the part of building image must place there is black
    img_aug = cv2.bitwise_and(img_aug, mask_inv)
    # place building image
    img_aug = cv2.bitwise_or(img_warp, img_aug)
    # show result!
    cv2.imshow('img', img_aug)

    cv2.waitKey(1)
