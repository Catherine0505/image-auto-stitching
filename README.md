Name: Catherine Gai

SID: 3034712396

Email: catherine_gai@berkeley.edu

Link to project report website: [Here](https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj4B/cs194-26-aay/catherine_gai_proj4b/Project%2004B.html)



This folder contains $10$ functional python files: "harris.py", "non_max_suppression.py", "descriptor_extraction.py", "feature_matching.py", "ransac.py", "main.py", "compute_projection.py", "warp_image.py", "define_features.py", "mosaic.py". 

Among these files, "compute_projection.py", "warp_image.py" and "define_features.py" are from the previous part, and function just the same is part (a). 

"mosaic.py" is slightly changed since the images we choose to warp are different from part (a). But if the reader follows the prompt, desired output will be generated. 



The folder also contains extra image files: "mosaic3_left.jpeg" (left view of the nighttime Berkeley Bay), "mosaic3_right.jpeg" (right view of the nighttime Berkeley Bay); "mosaic4_left.jpeg" (left view of MLK), "mosaic4_right.jpeg" (right view of MLK); "mosaic7_left.jpeg" (left view of Zellerbach Hall), "mosaic7_right.jpeg" (right view of Zellerbach Hall). 

"mosaic3_left.jpeg" is used during both mosaicing and visualization of feature generation and non-max suppression.



Furthermore, the folder contains several ".csv" files: "left_im_pts_3.csv", "right_im_pts_3.csv" for creating mosaic of the night Berkeley Bay; """left_im_pts_4.csv", "right_im_pts_4.csv" for creating mosaic of MLK; "left_im_pts_5.csv", "right_im_pts_5.csv" for creating mosaic of Zellerbach Hall. 



**harris.py:**

This python file contains all functions and commands that allows users to compute coarse feature points and visualize them on a certain image: "mosaic3_left.jpeg". 

* get_harris_corners(*params*): Given an image, the function computes corner strength on all pixels, and pick a set of coarse feature points, discarding those that are within 20 pixels from the edges (to facilitate later feature descriptor extraction). It returns a matrix of the same size as the input image, containing corner strength on each pixel, as well as the set of coarse feature points. 
* dist2(*params*): takes in two matrices $A, B$ of dimension: $M \times N$ and $L \times N$, the function computes the distance between each row of $A$ and each row of $B$. The returned matrix $C$ is of dimension $M \times L$, where $C_{i, j}$ denotes the squared Euclidean distance between the $i^{\text{th}}$ row of $A$ and the $j^{\text{th}}$ row of $B$. 

To visualize a set of coarse feature points, run `python harris.py` , this will give you the calculated coarse feature points on "mosaic3_left.py". 



**non_max_suppression.py:**

This python file contains function and commands that implements non-max suppression to pick the top $n$ feature points from the image, and visualize it on a certain image: "mosaic3_left.jpeg". 

* non_max_suppression(*params*): Given the corner strength of all pixels in the image as well as coarse feature points, the function picks top $n$ feature points ($n$ is determined by max_pts from the user, default to be 500). 

To visualize the refined feature points, run `python harris.py` , this will give you the top 500 feature points on "mosaic3_left.py". 



**descriptor_extraction.py:**

This python file contains a function that generates feature descriptors from a set of feature points. 

* extract_descriptor(*parmas*): Given an image, a set of feature points, the function extracts a feature descriptor around each point. The size of descriptor is determined by patch_height, patch_width and resize_ratio: $\text{descriptor height} = \frac{\text{patch height}}{\text{resize ratio}}$, $\text{descriptor width} = \frac{\text{patch width}}{\text{resize ratio}}$. 



**feature_matching.py:**

This python file contains a function that matches feature points between two images. 

* match_feature(*params*): Given feature point locations on two images and their corresponding feature descriptors, the function returns two numpy arrays that contain matched feature points in image 1 and image 2 respectively. If indexed into the same location within the array, a pair of matching feature points can be extracted. 



**ransac.py:**

This python file contains a function that computes desired affine transformation matrix between two images. 

* ransac(*params*): Given matched feature points of image 1 and image 2, the function calculates desired homography within a desginated number of iterations (indicated by "max_iter") and inlier tolerance (indicated by "threshold"). 

The function contains an interactive prompt. After going through sufficient number of sampling-validation cycle, the function will output the best homography that generates the most number of inliers. If user is satisfied with the result, type "Y" after the question and the final affine transformation matrix will be calculated with those inliers. Otherwise, type "N", specify the number of iterations, and the new inlier threshold, and the function will return back to the sampling-validation cycle. 



**main.py:**

This python file contains commands that produce three groups of mosaics, each with feathered and unfeathered results. The first group of mosaic is calculated using left and right view of the night Berkeley Bay; the second group of mosaic is calculated using left and right view of MLK; the third group of mosaic is calcualted using left and right view of Zellerbach Hall.  

