# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv2.cv2 import threshold
from scipy import linalg
import sys

'''
In this project you are asked to implement image stitching procedure
- Find out detectors and their description
- Match the detections between two images
- Compute homography and filters the outliers
- Apply projection to stitch the image s
'''


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = (int(cv2.__version__[0]) == 3)

    def findHomographyMatrix(self, x1, y1, x2, y2):
        A = np.array([[]])
        for i in range(4):
            a = np.array([[-x1[i], -y1[i], -1, 0, 0, 0, x2[i] * x1[i], x2[i] * y1[i], x2[i]]])
            b = np.array([[0, 0, 0, -x1[i], -y1[i], -1, y2[i] * x1[i], y2[i] * y1[i], y2[i]]])
            temp = np.concatenate((a, b))
            if i == 0:
                A = np.copy(temp)
            else:
                A = np.concatenate((A, temp))
        l, v, r = np.linalg.svd(A)
        # we want to get the eigen-vector that corresponds to the smallest eigen-value (closest to zero)
        return r[-1]

    def findHomography(self, ptsA, ptsB, threshold, maxIter=500):
        def reprojError(x, y):
            return \
                np.linalg.norm(np.dot(h, np.append(x, 1)) - np.append(y, 1)) + \
                np.linalg.norm(np.dot(h_inv, np.append(y, 1)) - np.append(x, 1))

        N = len(ptsA)
        indices = range(N)
        status = [0 for i in range(len(ptsA))]
        # np.random.shuffle(indices)
        index = 0
        best_h = np.random.random((3, 3))
        h = best_h
        h_inv = linalg.pinv(h)
        min_error = sum(reprojError(ptsA[i], ptsB[i]) for i in range(N)) / (2 * N)

        for j in range(maxIter):
            arr = np.random.randint(0, N - 1, 4)
            xy1 = ptsA[arr]
            xy2 = ptsB[arr]

            # h, _= cv2.findHomography(ptsA[arr], ptsB[arr], cv2.RANSAC, threshold)
            h = self.findHomographyMatrix(xy1[:, 0], xy1[:, 1], xy2[:, 0], xy2[:, 1])
            h = np.reshape(h, (3, 3))
            h_inv = linalg.pinv(h)
            reprojection_error = sum(reprojError(ptsA[i], ptsB[i]) for i in range(N)) / N
            # print reprojection_error

            if reprojection_error < min_error:
                best_h, min_error = h, reprojection_error
                # threshold = (max(reprojError(ptsA[i], ptsB[i]) for i in range(N))+min(reprojError(ptsA[i], ptsB[i]) for i in range(N)))/2

        status = [reprojError(ptsA[i], ptsB[i]) < min_error * 2 for i in range(N)]

        return (best_h, status)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        matches = self.matchKeypoints(featuresA, featuresB)

        M = self.computeHomography(kpsA, kpsB, matches, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    # def detectAndDescribe(self, image):
    #     # convert the image to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     # check to see if we are using OpenCV 3.X
    #     if self.isv3:
    #         # detect and extract features from the image
    #         descriptor = cv2.xfeatures2d.SIFT_create()
    #         (kps, features) = descriptor.detectAndCompute(image, None)
    #
    #     # otherwise, we are using OpenCV 2.4.X
    #     else:
    #         # detect keypoints in the image
    #         detector = cv2.FeatureDetector_create("SIFT")
    #         kps = detector.detect(gray)
    #
    #         # extract features from the image
    #         extractor = cv2.DescriptorExtractor_create("SIFT")
    #         (kps, features) = extractor.compute(gray, kps)
    #
    #     # convert the keypoints from KeyPoint objects to NumPy
    #     # arrays
    #     kps = np.float32([kp.pt for kp in kps])
    #
    #     # return a tuple of keypoints and features
    #     return (kps, features)

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, featuresA, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.match(featuresA, featuresB)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if True:
                matches.append((m.trainIdx, m.queryIdx))

        return matches

    def computeHomography(self, kpsA, kpsB, matches, reprojThresh):
        # print kpsA
        # print kpsB
        # print matches
        # print reprojThresh
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points

            # (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            # ptsA, ptsB are lists of the same size. index-correspondness
            (H, status) = self.findHomography(ptsA, ptsB, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


imageA = cv2.imread('A.jpg')
imageB = cv2.imread('B.jpg')

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

plt.figure()
plt.imshow(imageA)
plt.figure()
plt.imshow(imageB)
plt.figure()
plt.imshow(vis)
plt.figure()
plt.imshow(result)
plt.show()
