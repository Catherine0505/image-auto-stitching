import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio
import scipy
import scipy.spatial as spatial
import scipy.interpolate as interpolate
from compute_projection import computeH
from define_features import get_points

def warpImage(im, H, shape):
    """
    Warp an image to another projection plane. Transformation is specified by H.
    Output shape is specified by shape (height * width). In order to account for
    translation, the output shape may not be the same as the input image shape.
    :param im: image to be warped.
    :param H: the reversed projective transformation from current projection
        plane of im to another projection plane.
    :param shape: output shape.
    """
    height = im.shape[0]
    width = im.shape[1]
    channel = 3

    # Handle gray-scale images.
    if len(im.shape) == 2:
        im1 = np.expand_dims(im, axis = 2)
        channel = 1
    elif im.shape[2] == 1:
        channel = 1

    # Create meshgrid for later interpolation
    x = np.arange(height)
    y = np.arange(width)

    output = []

    for i in range(channel):
        # Create an empty matrix that records the projection result.
        output_channel = np.zeros((shape[0], shape[1]))
        im_channel = im[:, :, i]
        f_im = interpolate.RectBivariateSpline(x, y, im_channel)
        output_rr, output_cc = np.meshgrid(np.arange(shape[0]),
            np.arange(shape[1]))
        # Generate point locations on the resulting image.
        indices = np.stack((output_rr, output_cc, np.ones(output_cc.shape)),
            axis = -1)
        indices_flatten = indices.reshape((-1, 3))
        # Find the original point locations on the input image using reversed
        # projection matrix.
        im_indices = np.dot(H, indices_flatten.T)
        im_rr, im_cc = im_indices[0, :] / im_indices[2,:], \
            im_indices[1, :] / im_indices[2, :]
        # Eliminate the points that are out of boundary.
        mask_rr = (im_rr >= 0) & (im_rr < height)
        mask_cc = (im_cc >= 0) & (im_cc < width)
        mask = mask_rr * mask_cc
        output_values = f_im(im_rr, im_cc, grid = False) * mask
        output.append(output_values.reshape((shape[0], shape[1]), order = "F"))

    return np.dstack(output)


if __name__ == "__main__":
    im = skio.imread("left.jpeg")
    print("First Image Rectify: Use already existing features? [Y/N]")
    use_features = input()
    if use_features == "Y":
        im_pts = np.loadtxt("im_pts.csv", delimiter = ",")
        standard_pts = np.loadtxt("standard_pts.csv", delimiter = ",")
        im_pts_2 = np.loadtxt("rectify3_pts.csv", delimiter = ",")
        standard_pts_2 = np.loadtxt("standard3_pts.csv", delimiter = ",")
    else:
        im_pts = get_points(im, 8)
        standard_pts = get_points(im, 8)
    # First step of rectification: transform the image from upward view to
    # eyelevel view.
    H = computeH(standard_pts, im_pts)
    output = warpImage(im, H, [im.shape[0], im.shape[1]])
    skio.imsave("rectified_left.jpeg", output / 255)

    # Second step of rectification is based on results from the first step.
    im = skio.imread("rectified_left.jpeg")
    if use_features == "N":
        im_pts_2 = get_points(im, 8)
        standard_pts_2 = get_points(im, 8)
    # Second step of rectification: transform the image from side view to frontal
    # view.
    H = computeH(standard_pts_2, im_pts_2)
    output = warpImage(im, H, [im.shape[0], im.shape[1]])
    skio.imshow(output / 255)
    skio.show()

    if use_features == "N":
        print("First Image Rectify: Save image features or not? [Y/N]")
        save_bool = input()
        if save_bool == "Y":
            np.savetxt("im_pts.csv", im_pts, delimiter=",")
            np.savetxt("standard_pts.csv", standard_pts, delimiter=",")
            np.savetxt("rectify3_pts.csv", im_pts_2, delimiter = ",")
            np.savetxt("standard3_pts.csv", standard_pts_2, delimiter = ",")
    skio.imsave("rectified_left_further.jpeg", output / 255)


    im = skio.imread("mosaic2_right.jpeg")
    print("Second Image Rectify: Use already existing features? [Y/N]")
    use_features = input()
    if use_features == "Y":
        im_pts = np.loadtxt("rectify2_pts.csv", delimiter = ",")
        standard_pts = np.loadtxt("standard2_pts.csv", delimiter = ",")
        im_pts_2 = np.loadtxt("rectify4_pts.csv", delimiter = ",")
        standard_pts_2 = np.loadtxt("standard4_pts.csv", delimiter = ",")
    else:
        im_pts = get_points(im, 8)
        standard_pts = get_points(im, 8)
    # First step of rectification: transform the image from upward view to
    # eyelevel view.
    H = computeH(standard_pts, im_pts)
    output = warpImage(im, H, [im.shape[0], im.shape[1]])
    skio.imsave("rectified_right.jpeg", output / 255)

    # Second step of rectification is based on results from the first step.
    im = skio.imread("rectified_right.jpeg")
    # Fine tune the content position in source image to optimize results.
    im_shifted = np.zeros((im.shape[0], im.shape[1] + 200, 3))
    im_shifted[:, -im.shape[1]:] = im
    if use_features == "N":
        im_pts_2 = get_points(im, 8)
        standard_pts_2 = get_points(im, 8)
    im_pts_2_shifted = np.array(im_pts_2)
    standard_pts_2_shifted = np.array(standard_pts_2)
    # Second step of rectification: transform the image from side view to frontal
    # view.
    H = computeH(standard_pts_2, im_pts_2)
    # Output shape is elongated to include more rectified scenes. 
    output = warpImage(im_shifted, H, [im.shape[0], im.shape[1] + 800])
    skio.imshow(output / 255)
    skio.show()

    if use_features == "N":
        print("Second Image Rectify: Save image features or not? [Y/N]")
        save_bool = input()
        if save_bool == "Y":
            np.savetxt("rectify2_pts.csv", im_pts, delimiter=",")
            np.savetxt("standard2_pts.csv", standard_pts, delimiter=",")
            np.savetxt("rectify4_pts.csv", im_pts_2, delimiter = ",")
            np.savetxt("standard4_pts.csv", standard_pts_2, delimiter = ",")
    skio.imsave("rectified_right_further.jpeg", output / 255)
