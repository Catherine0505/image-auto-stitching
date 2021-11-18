import numpy as np
from skimage.feature import corner_harris, peak_local_max
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import resize
from harris import get_harris_corners
from harris import dist2
from descriptor_extraction import extract_descriptor
from feature_matching import match_feature
from compute_projection import computeH

def ransac(im1_coords, im2_coords, max_iter = 500, threshold = 4):
    """
    Implementation of RANSAC algorithm to find the affine transformation matrix
    between from image 1 to image 2.
    :param im1_coords: Feature points in image 1 that have valid matchings.
    :param im2_coords: Feature points in image 2 that have valid matchings.
    :param max_iter: Maximum number of iterations of homography-validation cycle.
        Default value: 500
    :param threshold: Constraint on inliers. A point would be considered as an
        inlier if the squared distance between its transformed coordinate and
        its matching feature location in image 2 is less than 4.
    """
    # Initialize number of best matches to keep track of the best affine
    # transformation matrix computer so far.
    # Initialize best_im1_coords_inlier and best_im2_coords_inlier to record
    # inlier points in image 1 (resp. image 2) corresponding to the best affine
    # transformation matrix.
    best_num_matches = 0
    best_im1_coords_inlier = None
    best_im2_coords_inlier = None

    for i in range(max_iter):
        # Choose four points randomly from image 1 and image 2 to compute a
        # candidate affine transformation matrix.
        indices = np.random.choice(im1_coords.shape[1], size = 4, replace = False)
        im1_pts = im1_coords[:, indices]
        im2_pts = im2_coords[:, indices]
        H = computeH(im1_pts.T, im2_pts.T)

        # Transform feature points in image 1 according to the computer affine
        # transformation matrix.
        im1_coords_add1 = np.concatenate((im1_coords,
            np.ones((1, im1_coords.shape[1]))))
        im1_coords_trans = np.dot(H, im1_coords_add1)
        im1_coords_trans = (im1_coords_trans / im1_coords_trans[2])[:2]

        # Compute the distance between each transformed feature location and
        # target feature location. Discard the points that are too far away.
        dist = np.sum((im1_coords_trans - im2_coords) ** 2, axis = 0)
        num_matches = np.sum((dist < threshold).astype(int))
        im1_coords_inlier = im1_coords[:, dist < threshold]
        im2_coords_inlier = im2_coords[:, dist < threshold]

        # Updata tracking records if necessary.
        if num_matches > best_num_matches:
            best_im1_coords_inlier = im1_coords_inlier
            best_im2_coords_inlier = im2_coords_inlier
            best_num_matches = num_matches

    # Interactive interface. User will decide whether to continue the sampling
    # and calculation process.
    print("Current best number of matches: ", best_num_matches)
    print("Are you satisfied? [Y/N]")
    satisfied = input()
    if satisfied == "N":
        print("How many more iterations do you want to try?")
        max_iter = int(input())
        print("What threshold do you want to set?")
        threshold = float(input())
        print("Going back...")
        ransac(im1_coords, im2_coords, max_iter = max_iter, threshold = threshold)
    else:
        best_H = computeH(best_im1_coords_inlier.T, best_im2_coords_inlier.T)
        return best_H
