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
from warp_image import warpImage


# left_im = skio.imread("left.jpeg")
# right_im = skio.imread("right.jpeg")
# print("First Mosaic: Use already existing features? [Y/N]")
# use_features = input()
# if use_features == "Y":
#     left_im_pts = np.loadtxt("left_im_pts.csv", delimiter = ",")
#     right_im_pts = np.loadtxt("right_im_pts.csv", delimiter = ",")
# else:
#     left_im_pts = get_points(left_im, 12)
#     right_im_pts = get_points(right_im, 12)
#
# # The standard point locations are shifted, since the standard image is stacked
# # on the right of the resulting image. Therefore, all point correspondences will
# # be shifted right by some value.
# right_im_pts_shifted = np.array(right_im_pts)
# right_im_pts_shifted[:, 1] = right_im_pts_shifted[:, 1] + 346
# result = np.zeros((left_im.shape[0], 346 + left_im.shape[1], 3))
# H = computeH(right_im_pts_shifted, left_im_pts)
# output = warpImage(left_im, H, [left_im.shape[0], 346 + left_im.shape[1]])
#
# # Generate feathered result.
# result[:, :346] = output[:, :346]
# result[:, -980:] = right_im[:, -980:]
# # Create an alpha mask around the boundary to do image feathering.
# alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
#     (left_im.shape[0], 1))
# alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
# result[:, 346:346 + 20] = alpha_mask * right_im[:, :20] + \
#     (1 - alpha_mask) * output[:, 346:346 + 20]
# skio.imshow(result / 255)
# skio.show()
# skio.imsave("mosaic1_masked.jpeg", result / 255)
#
# # Generate unfeathered result.
# unmasked_result = np.array(output)
# unmasked_result[:, -1000:] = right_im
# skio.imshow(unmasked_result / 255)
# skio.show()
# skio.imsave("mosaic1_unmasked.jpeg", unmasked_result / 255)
#
# if use_features == "N":
#     print("First Mosaic: Save image features or not? [Y/N]")
#     save_bool = input()
#     if save_bool == "Y":
#         np.savetxt("left_im_pts.csv", left_im_pts, delimiter=",")
#         np.savetxt("right_im_pts.csv", right_im_pts, delimiter=",")
#
#
# left_im = skio.imread("right.jpeg")
# right_im = skio.imread("mosaic2_right.jpeg")
# print("Second Mosaic: Use already existing features? [Y/N]")
# use_features = input()
# if use_features == "Y":
#     left_im_pts = np.loadtxt("left_im_pts_2.csv", delimiter = ",")
#     right_im_pts = np.loadtxt("right_im_pts_2.csv", delimiter = ",")
# else:
#     left_im_pts = get_points(left_im, 12)
#     right_im_pts = get_points(right_im, 12)
#
# result = np.zeros((left_im.shape[0], 368 + left_im.shape[1], 3))
# H = computeH(left_im_pts, right_im_pts)
# output = warpImage(right_im, H, [left_im.shape[0], left_im.shape[1] + 368])
#
# # Generate feathered result.
# result[:, :980] = left_im[:, :980]
# result[:, -368:] = output[:, -368:]
# # Create an alpha mask around the boundary to do image feathering.
# alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
#     (left_im.shape[0], 1))
# alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
# result[:, 980:1000] = alpha_mask * output[:, -388:-368] + \
#     (1 - alpha_mask) * left_im[:, 980:1000]
# skio.imshow(result / 255)
# skio.show()
# skio.imsave("mosaic2_masked.jpeg", result / 255)
#
# # Generate unfeathered result.
# unmasked_result = np.array(output)
# unmasked_result[:, :1000] = left_im
# skio.imshow(unmasked_result / 255)
# skio.show()
# skio.imsave("mosaic2_unmasked.jpeg", unmasked_result / 255)
#
# if use_features == "N":
#     print("Second Mosaic: Save image features or not? [Y/N]")
#     save_bool = input()
#     if save_bool == "Y":
#         np.savetxt("left_im_pts_2.csv", left_im_pts, delimiter=",")
#         np.savetxt("right_im_pts_2.csv", right_im_pts, delimiter=",")


left_im = skio.imread("mosaic3_left.jpeg")
right_im = skio.imread("mosaic3_right.jpeg")
print("Third Mosaic: Use already existing features? [Y/N]")
use_features = input()
if use_features == "Y":
    left_im_pts = np.loadtxt("left_im_pts_3.csv", delimiter = ",")
    right_im_pts = np.loadtxt("right_im_pts_3.csv", delimiter = ",")
else:
    left_im_pts = get_points(left_im, 8)
    right_im_pts = get_points(right_im, 8)

result = np.zeros((left_im.shape[0], 373 + left_im.shape[1], 3))
H = computeH(left_im_pts, right_im_pts)
output = warpImage(right_im, H, [left_im.shape[0], left_im.shape[1] + 373])

# Generate feathered result.
result[:, :680] = left_im[:, :680]
result[:, 700:] = output[:, 700:]
# Create an alpha mask around the boundary to do image feathering.
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (left_im.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
result[:, 680:700] = alpha_mask * output[:, 680:700] + \
    (1 - alpha_mask) * left_im[:, 680:700]
skio.imshow(result / 255)
skio.show()
skio.imsave("mosaic3_masked.jpeg", result / 255)

# Generate unfeathered result.
unmasked_result = np.array(output)
unmasked_result[:, :700] = left_im[:, :700]
skio.imshow(unmasked_result / 255)
skio.show()
skio.imsave("mosaic3_unmasked.jpeg", unmasked_result / 255)

if use_features == "N":
    print("Third Mosaic: Save image features or not? [Y/N]")
    save_bool = input()
    if save_bool == "Y":
        np.savetxt("left_im_pts_3.csv", left_im_pts, delimiter=",")
        np.savetxt("right_im_pts_3.csv", right_im_pts, delimiter=",")


left_im = skio.imread("mosaic4_left.jpeg")
right_im = skio.imread("mosaic4_right.jpeg")
print("Fourth Mosaic: Use already existing features? [Y/N]")
use_features = input()
if use_features == "Y":
    left_im_pts = np.loadtxt("left_im_pts_4.csv", delimiter = ",")
    right_im_pts = np.loadtxt("right_im_pts_4.csv", delimiter = ",")
else:
    left_im_pts = get_points(left_im, 8)
    right_im_pts = get_points(right_im, 8)

result = np.zeros((left_im.shape[0], right_im.shape[1] + left_im.shape[1], 3))
H = computeH(left_im_pts, right_im_pts)
output = warpImage(right_im, H, [left_im.shape[0], left_im.shape[1] + right_im.shape[1]])

# Generate feathered result.
result[:, :680] = left_im[:, :680]
result[:, 700:] = output[:, 700:]
# Create an alpha mask around the boundary to do image feathering.
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (left_im.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
result[:, 680:700] = alpha_mask * output[:, 680:700] + \
    (1 - alpha_mask) * left_im[:, 680:700]
skio.imshow(result / 255)
skio.show()
skio.imsave("mosaic4_masked.jpeg", result / 255)

# Generate unfeathered result.
unmasked_result = np.array(output)
unmasked_result[:, :700] = left_im[:, :700]
skio.imshow(unmasked_result / 255)
skio.show()
skio.imsave("mosaic4_unmasked.jpeg", unmasked_result / 255)

if use_features == "N":
    print("Fourth Mosaic: Save image features or not? [Y/N]")
    save_bool = input()
    if save_bool == "Y":
        np.savetxt("left_im_pts_4.csv", left_im_pts, delimiter=",")
        np.savetxt("right_im_pts_4.csv", right_im_pts, delimiter=",")


left_im = skio.imread("mosaic7_left.jpeg")
right_im = skio.imread("mosaic7_right.jpeg")
print("Fifth Mosaic: Use already existing features? [Y/N]")
use_features = input()
if use_features == "Y":
    left_im_pts = np.loadtxt("left_im_pts_5.csv", delimiter = ",")
    right_im_pts = np.loadtxt("right_im_pts_5.csv", delimiter = ",")
else:
    left_im_pts = get_points(left_im, 8)
    right_im_pts = get_points(right_im, 8)

result = np.zeros((left_im.shape[0], right_im.shape[1] + left_im.shape[1], 3))
H = computeH(left_im_pts, right_im_pts)
output = warpImage(right_im, H, [left_im.shape[0], left_im.shape[1] + right_im.shape[1]])

# Generate feathered result.
result[:, :680] = left_im[:, :680]
result[:, 700:] = output[:, 700:]
# Create an alpha mask around the boundary to do image feathering.
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (left_im.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
result[:, 680:700] = alpha_mask * output[:, 680:700] + \
    (1 - alpha_mask) * left_im[:, 680:700]
skio.imshow(result / 255)
skio.show()
skio.imsave("mosaic7_masked.jpeg", result / 255)

# Generate unfeathered result.
unmasked_result = np.array(output)
unmasked_result[:, :700] = left_im[:, :700]
skio.imshow(unmasked_result / 255)
skio.show()
skio.imsave("mosaic7_unmasked.jpeg", unmasked_result / 255)

if use_features == "N":
    print("Fifth Mosaic: Save image features or not? [Y/N]")
    save_bool = input()
    if save_bool == "Y":
        np.savetxt("left_im_pts_5.csv", left_im_pts, delimiter=",")
        np.savetxt("right_im_pts_5.csv", right_im_pts, delimiter=",")
