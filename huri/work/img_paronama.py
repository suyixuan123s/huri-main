# import the necessary packages
import numpy as np
import imutils
import cv2


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detect_and_describe(imageA)
        (kpsB, featuresB) = self.detect_and_describe(imageB)
        # match features between the two images
        M = self.match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren’t enough matched
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
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return result

    def detect_and_describe(self, image):
        # convert the image to grayscale
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        # img = cv2.drawKeypoints(image, kps, image)
        # cv2.imshow("sss", img)
        # cv2.waitKey(0)
        return (kps, features)

    def match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in raw_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe’s ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
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
                ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
                # return the visualization
        return vis


if __name__ == "__main__":
    # import the necessary packages
    import argparse
    import imutils
    import cv2
    from huri.core.common_import import fs

    c_pcd, pcds, imgs = fs.load_pickle("pcd_data")
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    imageA = imgs[1]
    imageB = imgs[0]
    imageA = imageA
    imageB = imageB
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # show the images
    cv2.imshow('Image A', imageA)
    cv2.imshow('Image B', imageB)
    cv2.imshow('Keypoint Matches', vis)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
