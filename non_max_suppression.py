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
from ransac import ransac


def non_max_suppression(h, coords, max_pts = 500, c_robust = 0.9):
    """
    Use non-max suppression algorithm to pick the top few feature points.
    :param h: Same size as the original image. Value at each point represents
        corner strength of that pixel.
    :param coords: pixel locations of coarse feature points.
    :param max_pts: Number of feature points to retain.
    :param c_robust: hyperparameter to suppress radius around a feature point.
    """
    coords_h = h[coords[0], coords[1]]
    coords_dist = dist2(coords.T, coords.T)
    ratio_matrix = np.outer(coords_h, (1 / coords_h))
    mask = ratio_matrix < c_robust
    coords_dist_masked = coords_dist * ratio_matrix
    coords_dist_nonzero = np.where(coords_dist_masked == 0, coords_dist_masked,
        np.inf)
    suppress_radius = np.min(coords_dist_nonzero, axis = 1)
    sort_indices = np.argsort(suppress_radius)
    candidate_indices = sort_indices[:max_pts]
    return coords[:, candidate_indices]


if __name__ == "__main__":
    # Compute non-max suppressed feature points of "mosaic3_left.jpeg".
    im = skio.imread("mosaic3_left.jpeg")
    h, coords = get_harris_corners(im[:, :, 0])
    print(coords.shape)
    coords_suppress = non_max_suppression(h, coords)
    print(coords_suppress.shape)
    plt.plot(coords_suppress[1], coords_suppress[0], linestyle = "None",
        marker = ".", markersize = 1, color = "r")
    plt.imshow(im)
    plt.show()
