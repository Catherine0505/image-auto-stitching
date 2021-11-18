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
from warp_image import warpImage
from non_max_suppression import non_max_suppression

# First automatic stitching: Night Berkeley Bay.
im1 = skio.imread("mosaic3_left.jpeg")
im2 = skio.imread("mosaic3_right.jpeg")

# Get the coarse feature points from image 1.
im1_h, im1_coords = get_harris_corners(im1[:, :, 0])
# Use non-max suppression to refine feature points to 2000.
im1_coords = non_max_suppression(im1_h, im1_coords, max_pts = 2000,
    c_robust = 0.8)
# Get the coarse feature points from image 2.
im2_h, im2_coords = get_harris_corners(im2[:, :, 0])
# Use non-max suppression to refine feature points to 2000.
im2_coords = non_max_suppression(im2_h, im2_coords, max_pts = 2000,
    c_robust = 0.8)

# Extract feature descriptor from image 1 and image 2.
im1_descriptor = extract_descriptor(im1[:, :, 0], im1_coords)
im2_descriptor = extract_descriptor(im2[:, :, 0], im2_coords)
# Filter out outlier points in image 1 and image 2 that can never be matched.
im1_coords_refined, im2_coords_refined = match_feature(im1_descriptor,
    im2_descriptor, im1_coords, im2_coords, 0.27)
print(im1_coords_refined.shape)
print(im2_coords_refined.shape)
# Show the tidied feature points that are ready to be passed on to affine
# transformation calculation.
plt.imshow(im1)
plt.plot(im1_coords_refined[1], im1_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()
plt.imshow(im2)
plt.plot(im2_coords_refined[1], im2_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()

# Use the RANSAC algorithm to calculate a desired affine transformation from
# image 1 to image 2.
H = ransac(im1_coords_refined, im2_coords_refined)
# Construct an image mosaic with feathering.
masked_result = np.zeros((im1.shape[0], im2.shape[1] + im1.shape[1], 3))
im2_warp = warpImage(im2, H, [im1.shape[0], im1.shape[1] + im2.shape[1]])
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (im1.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
masked_result[:, :im1.shape[1] - 20] = im1[:, :im1.shape[1] - 20]
masked_result[:, im1.shape[1]:] = im2_warp[:, im1.shape[1]:]
masked_result[:, im1.shape[1] - 20: im1.shape[1]] = alpha_mask * \
    im2_warp[:, im1.shape[1] - 20: im1.shape[1]] + \
    (1 - alpha_mask) * im1[:, im1.shape[1] - 20: im1.shape[1]]
skio.imshow(masked_result / 255)
skio.show()
# Construct an image mosaic without feathering.
unmasked_result = np.array(im2_warp)
unmasked_result[:, :im1.shape[1]] = im1
skio.imshow(unmasked_result / 255)
skio.show()


# Second automatic stitching: MLK.
im1 = skio.imread("mosaic4_left.jpeg")
im2 = skio.imread("mosaic4_right.jpeg")

# Get the coarse feature points from image 1.
im1_h, im1_coords = get_harris_corners(im1[:, :, 0])
# Use non-max suppression to refine feature points to 1000.
im1_coords = non_max_suppression(im1_h, im1_coords, max_pts = 1000,
    c_robust = 0.75)
# Get the coarse feature points from image 2.
im2_h, im2_coords = get_harris_corners(im2[:, :, 0])
# Use non-max suppression to refine feature points to 1000.
im2_coords = non_max_suppression(im2_h, im2_coords, max_pts = 1000,
    c_robust = 0.75)

# Extract feature descriptor from image 1 and image 2.
im1_descriptor = extract_descriptor(im1[:, :, 0], im1_coords)
im2_descriptor = extract_descriptor(im2[:, :, 0], im2_coords)
# Filter out outlier points in image 1 and image 2 that can never be matched.
im1_coords_refined, im2_coords_refined = match_feature(im1_descriptor,
    im2_descriptor, im1_coords, im2_coords, 0.27)
print(im1_coords_refined.shape)
print(im2_coords_refined.shape)
# Show the tidied feature points that are ready to be passed on to affine
# transformation calculation.
plt.imshow(im1)
plt.plot(im1_coords_refined[1], im1_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()
plt.imshow(im2)
plt.plot(im2_coords_refined[1], im2_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()

# Use the RANSAC algorithm to calculate a desired affine transformation from
# image 1 to image 2.
H = ransac(im1_coords_refined, im2_coords_refined, threshold = 4)
# Construct an image mosaic with feathering.
masked_result = np.zeros((im1.shape[0], im2.shape[1] + im1.shape[1], 3))
im2_warp = warpImage(im2, H, [im1.shape[0], im1.shape[1] + im2.shape[1]])
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (im1.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
masked_result[:, :im1.shape[1] - 20] = im1[:, :im1.shape[1] - 20]
masked_result[:, im1.shape[1]:] = im2_warp[:, im1.shape[1]:]
masked_result[:, im1.shape[1] - 20: im1.shape[1]] = alpha_mask * \
    im2_warp[:, im1.shape[1] - 20: im1.shape[1]] + \
    (1 - alpha_mask) * im1[:, im1.shape[1] - 20: im1.shape[1]]
skio.imshow(masked_result / 255)
skio.show()
# Construct an image mosaic without feathering.
unmasked_result = np.array(im2_warp)
unmasked_result[:, :im1.shape[1]] = im1
skio.imshow(unmasked_result / 255)
skio.show()


# Third automatic stitching: Zellerbach Hall.
im1 = skio.imread("mosaic7_left.jpeg")
im2 = skio.imread("mosaic7_right.jpeg")
# Get the coarse feature points from image 1.
im1_h, im1_coords = get_harris_corners(im1[:, :, 0])
# Use non-max suppression to refine feature points to 1500.
im1_coords = non_max_suppression(im1_h, im1_coords, max_pts = 1500,
    c_robust = 0.8)
# Get the coarse feature points from image 2.
im2_h, im2_coords = get_harris_corners(im2[:, :, 0])
# Use non-max suppression to refine feature points to 1500.
im2_coords = non_max_suppression(im2_h, im2_coords, max_pts = 1500,
    c_robust = 0.8)

# Extract feature descriptor from image 1 and image 2.
im1_descriptor = extract_descriptor(im1[:, :, 0], im1_coords)
im2_descriptor = extract_descriptor(im2[:, :, 0], im2_coords)
# Filter out outlier points in image 1 and image 2 that can never be matched.
im1_coords_refined, im2_coords_refined = match_feature(im1_descriptor,
    im2_descriptor, im1_coords, im2_coords, 0.27)
print(im1_coords_refined.shape)
print(im2_coords_refined.shape)
# Show the tidied feature points that are ready to be passed on to affine
# transformation calculation.
plt.imshow(im1)
plt.plot(im1_coords_refined[1], im1_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()
plt.imshow(im2)
plt.plot(im2_coords_refined[1], im2_coords_refined[0],
    linestyle = "None", marker = ".", markersize = 3, color = "r")
plt.show()

# Use the RANSAC algorithm to calculate a desired affine transformation from
# image 1 to image 2.
H = ransac(im1_coords_refined, im2_coords_refined, threshold = 4)
# Construct an image mosaic with feathering.
masked_result = np.zeros((im1.shape[0], im2.shape[1] + im1.shape[1], 3))
im2_warp = warpImage(im2, H, [im1.shape[0], im1.shape[1] + im2.shape[1]])
alpha_mask = np.tile(np.linspace(0, 1, 20, endpoint = True),
    (im1.shape[0], 1))
alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])
masked_result[:, :im1.shape[1] - 20] = im1[:, :im1.shape[1] - 20]
masked_result[:, im1.shape[1]:] = im2_warp[:, im1.shape[1]:]
masked_result[:, im1.shape[1] - 20: im1.shape[1]] = alpha_mask * \
    im2_warp[:, im1.shape[1] - 20: im1.shape[1]] + \
    (1 - alpha_mask) * im1[:, im1.shape[1] - 20: im1.shape[1]]
skio.imshow(masked_result / 255)
skio.show()
# Construct an image mosaic without feathering.
unmasked_result = np.array(im2_warp)
unmasked_result[:, :im1.shape[1]] = im1
skio.imshow(unmasked_result / 255)
skio.show()
