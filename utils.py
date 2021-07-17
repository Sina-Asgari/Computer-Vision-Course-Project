import numpy as np
import cv2
import random

def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr, thresh, num_iter=50, verbose=True):
    maxInliers = []
    finalH = None
    for j in range(num_iter):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        corr3 = corr[random.randrange(0, len(corr))]
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2, corr3, corr4))
        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        if verbose:
            print("Iter", str(j),"Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break

        # compute final homography with all inliers
        final_matrix = maxInliers[0]
        for t in maxInliers[1:]:
            final_matrix = np.vstack((final_matrix, t))
        finalH = calculateHomography(final_matrix)

    return finalH, maxInliers


def select_region(video_path='images/Video1.avi'):
    pts = np.zeros((4, 2), np.int16)
    counter = 0

    def mousePoints(event, x, y, flags, params):
        nonlocal counter
        if event == cv2.EVENT_LBUTTONDOWN:
            pts[counter] = x, y
            counter += 1

    vid = cv2.VideoCapture(video_path)
    _, img = vid.read()

    while True:

        for x, y in pts:
            cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2)  # x, y must be integer

        cv2.imshow('Input', img)
        cv2.setMouseCallback('Input', mousePoints)

        cv2.waitKey(1)
        if counter == 4:
            break

    return pts