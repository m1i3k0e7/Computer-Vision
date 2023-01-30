# Part 1. Harris Corner Detection
## A.a Gaussian Smooth
There are totally three functions in this part: 

(1) **Gaussian(sigma, x, y)**: This function will return the Gaussian value corresponding to σ = sigma, (x, y) = (x, y).

(2) **getKernel(size, sigma)**: This function will return a size × size Gaussian filter of σ = sigma.

(3) **gaussianSmoothing(img, size, sigma, read, imgName, srcPath, write, tarPath)**: This function will return an image convolved by a size × size Gaussian filter of σ = sigma. 

Convolutions in this function and following functions are all done by **scipy.signal.convolve2d()** built in a package named **scipy**.

There are the meanings of parameters of **gaussianSmoothing**:
- img: To input an image, you can simply read your image with **opencv** and pass it as img.
- read, imgName, srcPath: To input an image in another way, you can set read = 1 and input directory path and file name of your image to let the function read your image from local.
- write, tarPath: To save the image after convolution, you can set write = 1 and input the target path as tarPath to let the function save the image at tarPath.

## A.b Intensity Gradient (Sobel edge detection)
To compute the magnitude and direction of gradient of a blurred image, we predefined two sobel operators:

Hx = [[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]]

Hy = [[1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]]

There is one function in this part:
**intensityGradient(img, threshold, read, imgName, srcPath, write, tarPath)**: This function will

(1) respectively convolve your image with Hx and Hy to get gradients of image in  horizontal direction and vertical direction, namely, Ix and Iy. 

(2) Then, calculate the magnitude with $$magnitude = \sqrt{Ix^2 + Iy^2}$$ and filter out some weak magnitude with **threshold**.

(3) After getting magnitudes of image, calculate direction of each magnitude with **direction = arctan(Iy, Ix)**, **cv2.phase()** built in **opencv** is used to executing this step in this function. All direction will be classified into 8 types: 0&deg;, 45&deg; ..., 315&deg; and marked in different color.
 
(4) After finishing 3 steps above, Ix, Iy, magnitude and direction will be returned.
 
 ## A.c Structure Tensor
 There is one function in this part:
 **structureTensor(img, size, read, imgName, srcPath, write, tarPath)**: This function will

 (1) Get Ix and Iy mentioned in A.b and calculate 4 entries of structure tensor of each pixel:
 - Ix2 = Ix × Ix
 - Iy2 = Iy × Iy
 - IxIy = Ix × Iy
 - IyIx = Iy × Ix

define h, w = height, width of image, 4 entries mentioned above will be saved in four (h - size/2) × (w - size / 2) matrixes respectively, in this function, the parameter named **size** stands for the size of shifting window used to calculate the eigenvalues of image.

(2) Next, define one as a size × size matrix with all entries = 1, we can conveniently convolve Ix, Iy, IxIy and IyIx with one to get the sum of each entry in a shifting window centered at each pixel in img.

(3) Then, we can calculate **trace = Ix2 + Iy2** and **determinant = Ix2 × Iy2 - IxIy × IyIx** of each structure tensor and get the corner response R (i.e. smaller eigenvalue) of each pixel with $$R = determinant - 0.04 ×  trace^2$$

(4) After finishing 3 steps above, smaller eigenvalue of each pixel will be returned.

## A.d Non-maximal Suppression
there is one function in this part:

**nonMaximalSuppression(img, oriImg, threshold, misDis)**: There are the meanings of parameters:

- img: Smaller eigenvalues obtained from A.c
- oriImg: The unprocessed image that we want to mark corner in it.
- threshold: The threshold used to filter out weak eigenvalues.
- minDis: The minimum distance from a corner to another corner.

This function will:

(1) Filter out pixels whose eigenvalue is smaller than threshold and add all remaining pixels into a list L.

(2) Enter a while loop. In each iteration of while loop, the pixel with maximum eigenvalue in L will be extracted and marked on the oriImg. Then, pixels whose distance from the pixel with maximum eigenvalue is smaller than misDis will be removed from L. Not until L is empty will the while loop terminate. 

(3) After finishing 2 steps above, oriImg with marked corners will be returned.


# Part 2. SIFT interest point detection and matching
## A. SIFT interest point detection

there is one function in this part:

**interestPointDetection(img1, img2, nfeatures, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)**: There are the meanings of parameters:

- nfeatures:  The number of best features to retain.
- nOctaveLayers: The number of layers in each octave.
- contrastThreshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
- edgeThreshold: The threshold used to filter out edge-like features.
- sigma: The sigma of the Gaussian applied to the input image at the octave = #0.

This function will:

(1) Use SIFT detector to extract interest points from img1 and img2 by calling **cv2.xfeatures2d.SIFT_create()** and **sift.detectAndCompute()** built in **openCV**.

(2) Call **cv2.drawKeypoints()** built in **openCV** to draw interest point extracted in (1) on img1 and img2.

(3) Concatenate img1 and img2 into concat and return concat.

## B. SIFT feature matching

There are two function in this part: 

**oneNN_FeatureMatching(img1, img2, kps1, kps2, fea1, fea2, threshold)**: There are the meanings of parameters:

- kps1, kps2: Two lists of the interest points extracted from img1 and img2 in part A.
- fea1, fea2: Two lists of the corresponding features for all the interest point in kps1 and kps2.
- threshold: The threshold to filter out all the pairs of matched interest points with weak similarity after comparing the similarity between all the pairs between interest points in kps1 and kps2.

This function will:

(1) Declare a list of length = length of fea2 filled with [∞, -1] to record the point matched with each point kps2[i] in kps2 and the distance between them.

(2) For each point kps1[i] in kps1, calculate its distance from each point kps2[j] in kps2 and match kps1[i] with the interest point kps2[k] having the shortest distance from kps1[i] among all the points in kps2 and match kps1[i] with kps2[k] if distance between them is smaller than threshold. 

Note that each point in kps2 can be matched with at most one points in kps1. Thus, if kps2[k] has been matched with another point kps1[l] in kps1 previously, we should compare the distance between kps1[i] and kps2[k] and the distance between kps1[l] and kps2[k] to determine which point to be matched with kps2[k].

(3) Select all the pairs of interest points successfully matched and draw them on the concatenation of img1 and img2.

(4) Return the list of successfully matched points and the concatenated image.

**twoNN_FeatureMatching(img1, img2, kps1, kps2, fea1, fea2, threshold, ratio)**: This function is very similar to **oneNN_FeatureMatching** but still having some difference :

(1) In the step (2) of **oneNN_FeatureMatching**, we select the point kps2[j] whose distance to kps1[i] is the shortest, while in **twoNN_FeatureMatching**, we should additionally select the point kps2[k] whose distance to kps1[i] is the second shortest.

(2) the parameter ratio is used to determine whether to match kps1[i] with kps2[j] or not by checking $$ match_i,_j = |fea1[i] - fea2[j]| < ratio × (fea1[i] - fea2[k]) $$ 

If match_i,j is false, we cannot match kps1[i] with kps2[j] since the distances from kps1[i] to kps2[j] and from kps1[i] to kps2[k] is so similar that we cannot tell which point between kps2[j] and kps2[k] is the correct point should be matched with kps1[i].