#!/usr/bin/env python

# ---------------------------------
# CS472 - Assignment 4
# Date: 15/05/2022
# Adel Abbas, csdp1270
#
# Run:
#   python3 stitch.py
# ---------------------------------

import os
import cv2 as cv
import numpy as np

def readImage(filename):
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)


def writeImage(img, filename):
    cv.imwrite(filename, img)
    return None


def getConstraint(a: cv.KeyPoint, b: cv.KeyPoint):
    """
    Constructs a constraint matrix given 2 key points
    """
    ax, ay = a.pt
    bx, by = b.pt
    A = np.array(
        [[-ax, -ay, -1, 0, 0, 0, ax * bx, ay * bx, bx],
         [0, 0, 0, -ax, -ay, -1, ax * by, ay * by, by]]
    )
    return A


def findHomography(img1, img2, kp1, kp2, matches):
    """
    Computes the homography matrix using RANSAC.
    """
    NB_ITERS = 1000
    MAX_NB_INLIERS = 100
    maxInliersCount = 0
    bestInliers1 = None
    bestInliers2 = None
    iter = 0
    while maxInliersCount < MAX_NB_INLIERS and iter < NB_ITERS:
        # Choose random samples
        sampleIds = np.random.choice(matches.shape[0], size=4, replace=False)
        samples = matches[sampleIds, :]

        # Then construct the A matrix to be used in the DLT process
        constraints = tuple(getConstraint(kp1[a], kp2[b]) for a, b in samples)
        A = np.vstack(constraints)
        _, _, vh = np.linalg.svd(A)
        # The last row of vh corresponds to the eigenvector associated with
        # the smallest eigenvalue of A.T @ A
        h = vh[-1, :].reshape((3, 3))

        # Find the inliers
        matchIds1 = np.array(matches[:, 0], dtype=int)
        matchIds2 = np.array(matches[:, 1], dtype=int)
        inliers1, inliers2 = getInliers(
            np.array(kp1)[matchIds1], np.array(kp2)[matchIds2], h)
        count = len(inliers1)

        if count >= maxInliersCount:
            maxInliersCount = count
            bestInliers1 = inliers1
            bestInliers2 = inliers2
        iter += 1

    # Recompute h with the best set of inliers
    constraints = tuple(getConstraint(kp1, kp2)
                        for kp1, kp2 in zip(bestInliers1, bestInliers2))
    A = np.vstack(constraints)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1, :].reshape((3, 3))

    return h


def getInliers(kp1, kp2, h):
    """
    Finds the inlier key points pairs.
    """
    THRESHOLD = 600
    kp1_coordinates = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kp1])
    kp2_coordinates = np.array([list(kp.pt) for kp in kp2])

    prediction = kp1_coordinates @ h
    # Normalize coordinates [x, y, alpha] => [x, y, 1]
    prediction = prediction / prediction[:, 2][:, None]
    prediction = prediction[:, :2]
    dist = np.linalg.norm(prediction - kp2_coordinates, axis=1)
    inliersIds = np.argwhere(dist < THRESHOLD).flatten()
    return kp1[inliersIds], kp2[inliersIds]

def warpImages(img1, img2, h):
    height1, width1 = img1.shape
    height2, width2 = img2.shape
    # Find the corners of the new images
    srcImageCorners = np.array([[0, 0],
                                [0, height1],
                                [width1, height1],
                                [width1, 0]], dtype=np.float64)
    srcImageCorners = srcImageCorners.reshape(-1, 1, 2)

    destImageCorners = np.array([[0, 0],
                                 [0, height2],
                                 [width2, height2],
                                 [width2, 0]], dtype=np.float64)
    destImageCorners = destImageCorners.reshape(-1, 1, 2)

    # Translate the corners of the first image
    srcImageCorners = cv.perspectiveTransform(srcImageCorners, h)

    corners = np.concatenate(
        (srcImageCorners, destImageCorners), axis=0)

    # Compute the corners of the output image
    margin = 1
    x_min, y_min = np.int32(corners.min(axis=0).ravel() - margin)
    x_max, y_max = np.int32(corners.max(axis=0).ravel() + margin)
    shape = (x_max - x_min, y_max - y_min)
    x_min = -x_min
    y_min = -y_min

    # Homography translation matrix
    ht = np.array([[1, 0, x_min],
                   [0, 1, y_min],
                   [0, 0, 1]])

    # Warp the source image in the output image
    warp = cv.warpPerspective(img1, ht.dot(h), dsize=shape)

    warp[y_min: height1 + y_min, x_min: width1 + x_min] = img2
    return warp


def getTopMatches(distances, n):
    """
    Return a matrix with the indices of the
    top-N matches. This step is the slowest part of the entire 
    stichting process, as the sorting task is performed on a 
    relatively large matrix.
    """
    width = distances.shape[1]
    flat = distances.flatten()
    topN = flat.argsort()[:n]
    matches = np.zeros((n, 2), dtype=int)
    for k, i in enumerate(topN):
        x = i % width
        y = i // width
        matches[k] = [y, x]
    return matches


def computeDistanceMatrix(A, B):
    """
    In order to benefit from the efficiency of vectorized numpy computations,
    the distance matrix dist(A, B) is computed as :
    dist(A, B) = sqrt(dot(A, A) + dot(B, B) - 2 * dot(A, B))
    """
    M = A.shape[0]
    N = B.shape[0]
    AA = np.sum(A*A, axis=1).reshape((M, 1)) * np.ones((1, N))
    BB = np.sum(B*B, axis=1) * np.ones((M, 1))
    distances = AA + BB - 2 * A.dot(B.T)
    return np.sqrt(distances)


def stitch(img1, img2):
    print(" Computing SIFT features")
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(" Computing distance matrix")
    distances = computeDistanceMatrix(des1, des2)

    print(" Computing top n matches")
    n = 200
    matches = getTopMatches(distances, n)

    print(" Computing homography")
    h = findHomography(img1, img2, kp1, kp2, matches)

    print(" Warping images")
    stitched = warpImages(img1, img2, h)
    return stitched


def process(filename, filename2, data="./", out="./"):
    print("-------------------------")
    print(" Reading images")
    img1 = readImage(os.path.join(data, filename))
    img2 = readImage(os.path.join(data, filename2))
    res = stitch(img1, img2)
    writeImage(res, os.path.join(out, "stitch.jpg"))
    return None


def main():
    DIR = "./data"
    OUT = "./out"
    filename = "uttower_left.JPG"
    filename2 = "uttower_right.JPG"
    #filename = "1.JPG"
    #filename2 = "2.JPG"
    process(filename, filename2, data=DIR, out=OUT)

    return None


if __name__ == "__main__":
    main()
