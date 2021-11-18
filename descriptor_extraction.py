import numpy as np
from skimage.feature import corner_harris, peak_local_max
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import resize
from harris import get_harris_corners

def extract_descriptor(im, coords, patch_height = 40, patch_width = 40,
    resize_ratio = 5):
    """
    Extract feature descriptors from a image.
    Output shape: (num of feature points, patch_height // resize_ratio,
    patch_width // resize_ratio)
    :param im: Input image.
    :param coords: Coordinates of feature points in the image.
    :param patch_height: Height of patch to extract feature descriptors
        before subsampling.
    :param patch_width: Width of patch to extract feature descriptors
        before subsampling.
    :resize_ratio: The ratio of subsampling. Final descriptor size will be
        (patch_height // resize_ratio, patch_width // resize_ratio).
    """
    # Create an empty list to contain all feature descriptors.
    list = []

    # For each feature point, calculate the feature descriptor, and normalize it
    # to have mean 0 and std 1.
    for i in range(coords.shape[1]):
        loc = [coords[0][i], coords[1][i]]
        sub_im = im[loc[0] - patch_height // 2: loc[0] + patch_height // 2, \
            loc[1] - patch_width // 2: loc[1] + patch_width // 2]
        sub_im = resize(sub_im,
            (patch_height // resize_ratio, patch_width // resize_ratio),
            anti_aliasing = True)
        sub_im_avg = np.mean(sub_im)
        sub_im_std = np.std(sub_im)
        if sub_im_std == 0:
            sub_im = sub_im - sub_im_avg
        else:
            sub_im = (sub_im - sub_im_avg) / sub_im_std
        list.append(sub_im)
    list = np.array(list)
    print(list.shape)
    return list

if __name__ == "__main__":
    # Compute feature descriptors of "mosaic3_left.jpeg". Sanity check. 
    im = skio.imread("mosaic3_left.jpeg")
    _, coords = get_harris_corners(im[:, :, 0], num_corners = 532)
    print(coords.shape)
    sub_im_list = extract_descriptor(im[:, :, 0], coords)
    sub_im_list_flatten = sub_im_list.reshape((500, -1))
    print(sub_im_list_flatten.shape)
