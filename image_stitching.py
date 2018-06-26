# import the necessary packages
import scipy.signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal


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
        # result = cv2.warpPerspective(imageA, H,
        #                              (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result = self.my_warp_perspective(imageA,H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
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

    def my_warp_perspective(self,imageA, H,dest_shape):
        width,height = dest_shape
        res = np.zeros((height,width,3)).astype(imageA.dtype)
        a_rows,a_cols,_ = imageA.shape
        index_mat = np.zeros((a_rows,a_cols,2))

        x,y = np.meshgrid(range(a_cols),range(a_rows))
        o = np.ones(x.shape)
        src_vecs = np.array([np.reshape(x,[-1]),np.reshape(y,[-1]),np.reshape(o,[-1])])
        mul_sum = np.dot(H,src_vecs)
        dst_x = mul_sum[0,:] / mul_sum[2,:]
        dst_y = mul_sum[1,:] / mul_sum[2,:]
        dst_x = dst_x.reshape(a_rows,a_cols)
        dst_y = dst_y.reshape(a_rows,a_cols)

        dst_x = dst_x.astype(np.int)
        dst_y = dst_y.astype(np.int)
        ind = np.where((dst_x>=0) * (dst_y>=0))
        res[dst_y[ind],dst_x[ind]]=imageA[y[ind],x[ind]]
        # do conv with kernal of size 5
        # kernel = np.array([0.05,0.25,0.4,0.25,0.05])
        # res[:,:,0] = scipy.signal.convolve2d(res[:,:,0].reshape(height,width), kernel, 'same')
        # res[:,:,1] = scipy.signal.convolve2d(res[:,:,1].reshape(height,width), kernel, 'same')
        # res[:,:,2] = scipy.signal.convolve2d(res[:,:,2].reshape(height,width), kernel, 'same')

        # cv2.imshow('check', res)
        # cv2.waitKey()
        return res

    # def rgb_bilinear_interpolation(self,image):
    #     rows,cols = image.shape
    #
    # def rgb_b_inter_helper(self,x,y,points):
    #     """
    #
    #     :param x: the x value of the point to interpolate
    #     :param y: the y value of the point
    #     :param points: the neighbors points as array that each cell is list of [xi,yi,[r,g,b]]
    #     :return: the r,g,b value s of the point
    #     """
    #     s_points = sorted(points) # order points by x, then by y

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

    @classmethod
    def sum9(cls, mat):
        return np.sum(cls.shifted9(mat), axis=0)

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        keypoints, descriptors = self.describe(image, self.detect(image))
        keypoints = np.float32([kp.pt for kp in keypoints])
        return keypoints, descriptors

    def detect(self, image):
        """
        Harris Corner Detector
        :param image: RGB Matrix
        :return: list of cv2.KeyPoint
        """
        K = 0.05
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float) / 255.
        x_kernel = np.array([[-1, 0, 1]] * 3)
        Ix = scipy.signal.convolve2d(gray, x_kernel, 'same')
        Iy = scipy.signal.convolve2d(gray, x_kernel.T, 'same')

        MIxx = self.sum9(Ix ** 2)
        MIyy = self.sum9(Iy ** 2)
        MIxy = self.sum9(Ix * Iy)

        img_det = (MIxx * MIyy) - (MIxy * MIxy)
        img_trace = MIxx + MIyy

        img_corners = (img_det - K * img_trace ** 2) > 0
        np_keypoints = np.argwhere(img_corners)
        np_keypoints = [np_keypoints[i] for i in xrange(0, len(np_keypoints), 20)]
        keypoints = [cv2.KeyPoint(x, y, 30) for x, y in np_keypoints]
        return keypoints

    def describe(self, image, keypoints):
        if not self.isv3:
            raise RuntimeError("OpenCV version should be 3.X")
        sift = cv2.xfeatures2d.SIFT_create()
        matched_keypoints, descriptors = sift.compute(image, keypoints)
        return matched_keypoints, descriptors

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
