# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    @staticmethod
    def shifted9(mat):
        all_mat = [mat.copy() for _ in range(9)]
        all_mat[0][1:, 1:] = mat[:-1, :-1]
        all_mat[1][:, 1:] = mat[:, :-1]
        all_mat[2][:-1, 1:] = mat[1:, :-1]
        all_mat[3][1:, :] = mat[:-1, :]
        all_mat[4][:, :] = mat[:, :]
        all_mat[5][:-1, :] = mat[1:, :]
        all_mat[6][1:, :-1] = mat[:-1, 1:]
        all_mat[7][:, :-1] = mat[:, 1:]
        all_mat[8][:-1, :-1] = mat[1:, 1:]
        return all_mat

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        keypoints, descriptors = self.describe(image, self.detect(image))
        keypoints = np.float32([kp.pt for kp in keypoints])
        return keypoints, descriptors

    def detect(self, image):
        if not self.isv3:
            raise RuntimeError("OpenCV version should be 3.X")
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(image)
        return keypoints

    def describe(self, image, keypoints):
        if not self.isv3:
            raise RuntimeError("OpenCV version should be 3.X")
        sift = cv2.xfeatures2d.SIFT_create()
        matched_keypoints, descriptors = sift.compute(image, keypoints)
        return matched_keypoints, descriptors

    def matchKeypoints(self, featuresA, featuresB):
        # matches1 = self.danielMatcher(featuresA,featuresB, method = "crossRef")
        matches1 = self.danielMatcher(featuresA,featuresB, method = "bruteForce")
        matches = matches1.tolist()
        print("daniel finished")
        if False:
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
            # match_matches = (np.array(matches) == matches1)

        print("matches length " + str(len(matches)))

        return matches

    def danielMatcher(self, featuresA, featuresB ,method = "bruteForce", ptsA = None, ptsB = None):

        # for A find best match in B by euclidean distance
        if method == "bruteForce":
            if len(featuresA) < len(featuresB):
                matchesA, distA = self.findDist(featuresB, featuresA)
            else:
                matchesA, distA = self.findDist(featuresA, featuresB)
            matchesB = distB = []

        # crossRef from B to A
        else: # method == "crossRef":
            matchesA, distA = self.findDist(featuresA, featuresB)
            matchesB, distB = self.findDist(featuresB, featuresA)

        # find certainty by dividing with next best match
        matchList, certainty = self.find_matches(matchesA, matchesB, distA, distB, method)
        # remove matches by certainty
        matchList,_ = self.decide_matches(matchList, certainty, cert_threshold=1.5)
        # strengthen points by KNN
        return matchList

    def findDist(self, featA, featB, dist_threshold = 200000):
        len(featA)
        len(featB)
        mtcA = []
        distA = []
        for i in range(len(featA)):
            dist = [dist_threshold, dist_threshold]
            mtc = [-1, -1]
            for j in range(len(featB)):
                temp = np.sum(np.power(featA[i]-featB[j], 2))
                if temp < dist[0]:
                    dist[1] = dist[0]
                    mtc[1] = mtc[0]
                    dist[0] = temp
                    mtc[0] = j
                elif temp < dist[1]:
                    dist[1] = temp
                    mtc[1] = j
            mtcA.append(mtc)
            distA.append(dist)
        return mtcA, distA

    def decide_matches(self, matchList, certainty, cert_threshold=0.5):
        # find matches with certainty over threshold
        mask = np.array(certainty) < cert_threshold
        matchList = np.array(matchList)[mask]
        certainty = np.array(certainty)[mask]

        # sort matches
        # idx = np.argsort(certainty)
        # matchList = np.array(matchList)[idx]
        # certainty = np.array(certainty)[idx]

        return matchList, certainty

    def find_matches(self, matchesA, matchesB, distA, distB, method):
        matchList = []
        certaintyList = []
        # certainty = [-1, -1]
        if method == "crossRef":
            for i, match in enumerate(matchesA):
                j1 = match[0]
                # t = matchesB[j1][0]
                if i == matchesB[j1][0]:
                    matchList.append((i, j1))
                    certainty1 = distA[i][0] / distA[i][1]
                    certainty2 = distB[j1][0] / distB[j1][1]
                    certainty = np.sqrt(certainty1 * certainty2)
                    certaintyList.append(certainty)
        else:
            for i, match in enumerate(matchesA):
                j1 = match[0]
                matchList.append((i, j1))
                certaintyList.append(distA[i][0] / distA[i][1])
        return matchList, certaintyList


    def computeHomography(self, kpsA, kpsB, matches, reprojThresh):

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

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

# s = Stitcher()
# matchList = [(1,1), (2,2), (3,3), (4,4)]
# certainty = np.array([0,2,5,3])
# s.decide_matches(matchList,certainty, 2.5)


if True:
    imageA = cv2.imread('A.jpg')
    imageB = cv2.imread('B.jpg')

    # imageA = imageA[50:200, 250:500, :]
    # imageB = imageB[80:230,:250, :]
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    plt.figure()
    plt.imshow(imageA)
    plt.figure()
    plt.imshow(imageB)
    plt.figure()
    # plt.imshow(imageA1)
    # plt.figure()
    # plt.imshow(imageB1)
    # plt.figure()

    plt.imshow(vis)
    plt.figure()
    plt.imshow(result)
    plt.show()
