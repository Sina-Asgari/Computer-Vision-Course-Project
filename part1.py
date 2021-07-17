import numpy as np
import cv2
import random
from utils import *

# read images
img1 = cv2.imread('images/building1.jpg')
img2 = cv2.imread('images/building2.jpg')


estimation_thresh = 1

# create an instance to using orb detector
orb = cv2.ORB_create()

# using brute force matcher to matches features
bf = cv2.BFMatcher()

correspondenceList = []
if img1 is not None and img2 is not None:
    # detect keypoints and compute their descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    keypoints = [kp1, kp2]
    # matchs TWO descriptors for better matching
    matches = bf.knnMatch(des1, des2, k=2)

    print(f'Total number of matches: {len(matches)}')

    # apply ratio test to find better matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f'Total number of good matches: {len(good)}')

    # find corresponding keypoints for compute homography
    for match in good:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)
    # find homography between two images using ransac
    H, inliers = ransac(corrs, estimation_thresh)

    print("Final homography: ", H, sep='\n')
    print("Final inliers count: ", len(inliers))

    # draw 10 best matches
    matchImg = cv2.drawMatches(img1, kp1, img2, kp2, good[:10], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matchImg', matchImg)

    # for stitching to gather
    stitcher = cv2.Stitcher.create()
    status, stitched_image = stitcher.stitch([img1, img2])
    if status == cv2.STITCHER_OK:
        cv2.imshow('stitching', stitched_image)

    cv2.waitKey(0)
