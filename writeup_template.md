
### Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/hog2.png
[image3]: ./output_images/hog3.png
[image4]: ./output_images/detecttest.png
[image5]: ./output_images/detecttest_1.png
[image6]: ./output_images/heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for this step is contained in the second code cell of "./Vehicle_Detection.ipynb". I directly used the uitl funtions in the Udacity lessons for this part as it works well.I started by reading in all the `vehicle` and `non-vehicle` images and exploring different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 2. Choice of HOG parameters.

I tried various combinations of parameters and color spaces. Here is an example using the three different color spaces:`YCrCb`,`YUV`,`HLS` and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Here is an example using the same color spaces:`YCrCb` yet different HOG parameters:

* `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
* `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`
* `orientations=12`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`

![alt text][image3]
I tried various combinations of parameters and chose the HOG parameters that provided the maximum prediction accuracy on the test-set after training.

#### 3. Train a classifier using selected HOG features and color features.

I trained a linear SVM, in cell 8, using the features from the extract_features() function in cell 2 that stacks the HOG, spatial binning and color histogram features. These features are then scaled using StandardScaler() for training. The final parametrs that I used for extracting the features are: 

HOG orientations = 9 pix_per_cell = (8,8) cell_per_block = (2,2) hog_channel = 'ALL' spatial binning size = (32, 32) histogram bins = 32. 

The data are then split into training and test sets with 75% to 25% ratio. For tuning the C parameter of SVM classifier I used GridSearchCV function to search for the best combination. Between the cases I searched, the best comibnation was achieved by YUV colorspace and C=0.1 the gave %99 test set accuracy.

### Sliding Window Search
At first I used the slidng window function in cell 2 to create windows with defined size: `xy_window` and steps: `xy_overlap` on the image and then extracted the features of image in each image using function `search_windows` in cell 4. All the window detected as car will be recorded. This pipeline works well yet lack of efficiency.

Instead, I applied the function `find_cars`in cell 4.1 to detect image windows. In compare to above method, the hog feature of the entire image only need to be calculated once, which reduces the process time cost to aout one tenth. The result turned out to be as good as the first method.

In order to accalerate the process further, the search area is limited to lower half of the image through parameters: `ystart=360, ystop=650`. 

As the distance between the camera and other vehicles has a big influence on the detection rate, I need to search the image three times on three scales: 1, 1.5, 2. To do this, I modified the `find_cars`in cell 4.1 to `find_cars_multiscales` in cell 9.3, which take a list of scales and return window list and marked image.

![alt text][image5]

#### 2. Result of test images and methods to optimize the performance?

Ultimately I searched on Three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

#### 3. Filtered combined overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image6]

---

### Video Implementation

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

* In order to detect vehicles in the near and far away in the same time, the whole image need to be searched multiple times with different scales. This slows down the process.
* Small scales are needed for dectection of cars far away, yet they are more prone to create false positive. A well designed search method with combination of search area and scale may improve the performance.

