import numpy as np
from skimage.feature import corner_harris, peak_local_max
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import resize
from harris import get_harris_corners
from harris import dist2
from descriptor_extraction import extract_descriptor


def match_feature(im1_descriptor, im2_descriptor, im1_coords, im2_coords, threshold):
    """
    Match feature points in image 1 with points in image 2. Manage to only
    preserve valid matchings.
    :param im1_descriptor: Feature descriptors of image 1.
    :param im2_descriptor: Feature descriptors of image 2.
    :param im1_coords: Feature point locations in image 1.
    :param im2_coords: Feature point locations in image 2.
    :threshold: constraint on valid feature matchings. A matching is considered
        valid if and only if
        diff(best match) / diff(second best match) < threshold.
    """
    # Handle inputs. If input is a 3-D vector, flatten the 2-D descriptor for
    # each feature point.
    if len(im1_descriptor.shape) == 2:
        im1_descriptor_flatten = im1_descriptor
    else:
        im1_descriptor_flatten = im1_descriptor.reshape(im1_descriptor.shape[0], -1)

    if len(im2_descriptor.shape) == 2:
        im2_descriptor_flatten = im2_descriptor
    else:
        im2_descriptor_flatten = im2_descriptor.reshape(im2_descriptor.shape[0], -1)

    # Compute feature descriptor correlations between each feature point in
    # image 1 and each point in image 2.
    correlation = dist2(im1_descriptor_flatten, im2_descriptor_flatten)

    # Find the ratio between difference of best match and second best match for
    # each feature point in image 1.
    min_correlation_row = np.min(correlation, axis = 1)
    correlation_ratio = \
        min_correlation_row.reshape((len(min_correlation_row), 1)) / correlation
    correlation_sort = np.argsort(correlation_ratio, axis = 1)
    nn1_index = correlation_sort[:, -1]
    nn2_index = correlation_sort[:, -2]
    nn2_ratio = correlation_ratio[list(range(correlation.shape[0])), nn2_index]

    # Filter out the points whose best match and second best match are too
    # similar, since this indicates that there is likely to be no valid matching.
    mask = nn2_ratio < threshold
    im1_pts = np.arange(correlation.shape[0])[mask]
    im2_pts = nn1_index[mask]
    return im1_coords[:, im1_pts], im2_coords[:, im2_pts]
