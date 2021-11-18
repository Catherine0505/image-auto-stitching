import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio
import scipy

def computeH(im1_pts, im2_pts):
    """
    Compute the projective transformation matrix based on point correspondences.
    :param im1_pts: Feature points chosen for image 1.
    :param im2_pts: Feature points chosen for image 2.
    """
    im2_pts_reshape = np.reshape(im2_pts, im2_pts.shape[0] * im2_pts.shape[1])
    A = np.zeros((im1_pts.shape[0] * 2, 8))
    A[::2, [0, 1]] = im1_pts
    A[1::2, [3, 4]] = im1_pts
    A[::2, 2] = 1
    A[1::2, 5] = 1
    A[:, 6] = -np.repeat(im1_pts[:, 0], 2) * im2_pts_reshape
    A[:, 7] = -np.repeat(im1_pts[:, 1], 2) * im2_pts_reshape
    params = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, im2_pts_reshape))
    H = np.array([[params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1]])
    return H
