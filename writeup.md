# **Traffic Sign Recognition** 

## Writeup
For Udacity's Self-Driving Car NanoDegree, Term 1, Project 2

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/class_histogram.png "Visualization"
[image21]: ./writeup_images/preprocessed_img_a.png "Pre-Processing A"
[image22]: ./writeup_images/preprocessed_img_b.png "Pre-Processing B"
[image23]: ./writeup_images/preprocessed_img_c.png "Pre-Processing C"
[image3]: ./test_data_from_web/web_sample_1.png "Collage of German Traffic Signs"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is link to my [project code](https://github.com/AeroGeekDean/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) as a Jupyter Notebook.

### Data Set Summary & Exploration

The code for this step is contained in the **"Step 1: Dataset Summary & Exploration"** section of the Jupyter notebook.  

The basic statistics for the dataset were calculated using Numpy array functions, and are provided below:

* The size of training set is 34799 image samples
* The size of test set is 12630 image samples
* The shape of a traffic sign image is 32x32 pixels, with 3 layers of color (RGB).
* The number of unique classes/labels in the data set is 43

A histogram of the unique classes/labels distribution, for both the training and cross-validation sets, are shown below:

![Class distribution][image1]

We can see that there are skewness in the data, not all classes are evenly distributed in both training and cross-validation dataset. We should keep this in mind later when evaluating the performance of the classifier model.

For sanity check, some random samples of the training images were also plotted out to confirm they are indeed images of traffic signs. These are visible in the Jupyter notebook.

### Design and Test a Model Architecture

The code for this step is contained under the **"Step 2: Design and Test a Model Architecture"** section of the Jupyter notebook.

#### 1. Data Pre-processing

- Contrast Normalization

While looking at some of the training images during the Exploratory Data visualization stage earlier, it was noticed that often the images are dark and have poor contrast. Thus contrast normalization were performed on each image individually. This was accomplished by scaling each images constrast to be between the range of [0, 255], utilizing `'OpenCV2.normalize()'` function.

- Feature Normalization

Next, feature normalization were performed on the entire training set to center and scale the data (zero mean, unit variance), utilizing `'SkLearn.preprocessing.StandardScaler'` feature. Each (32x32x3) image pixel were reshaped into a (1x3072) long feature vector, the entire training set was normalized, then each sample is reshaped back to (32x32x3) image size.

**Important Note:** Scaling & mean-centering parameters were derived based ONLY on the training set. Later during model performance evaluation, these stored parameters were then applied onto the cross-validation and test data sets.

Here are a few examples of traffic sign images in its original 'raw' condition, re-contrasted, and feature normalized.

![pre-processed image A][image21]

![pre-processed image B][image22]

![pre-processed image C][image23]

**Notice** contrast adjustment improved image visual clarity TO THE HUMAN EYE/BRAIN, while feature normalization (across the training set) did the opposite!!

#### 2. Validation Strategy

The dataset provided has already been split out for training/cross-validation/testing. The split is about:

| Data Set | % |
|:-----:|:-----:|
| Training | 67.1 |
| Cross-Validation | 8.5 |
| Testing | 24.4 |

Another validation strategy is to utilize the [K-Fold Cross-Validation method](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation), unfortunately I ran out of time thus did not implement for this project... :\

#### 3. Model Architecture

The code for my final model is located in the **Model Architecture** section of the Jupyter notebook. 

I started with the LeNet architecture provided straight from the lecture, adjusting for the input and output sizes. This gave unacceptable performance over-fitting results, thus a **dropout layer** was added before each activation.

My final model consisted of the following:

| Model Layers |
|-----|
|
| **Layer 1:**<ul><li>**Convolution**: Input = 32x32x3. Filter = (5x5x3)x6. Output = 28x28x6.</li><li>**Dropout**</li><li>**Activation**: ReLU</li><li>**Pooling**: Input = 28x28x6. Filter = 2x2. Output = 14x14x6.</li></ul>
|
|**Layer 2:**<ul><li>**Convolution**: Input = 14x14x6. Filter = (5x5x6)x16. Output = 10x10x16.</li><li>**Dropout**</li><li>**Activation**: ReLU</li><li>**Pooling**: Input = 10x10x16. Filter = 2x2. Output = 5x5x16.</li></ul>
|
|**Flatten:** Input = 5x5x16. Output = 400.
|
|**Layer 3:**<ul><li>**Fully Connected**: Input = 400. Output = 120.</li><li>**Dropout**</li><li>**Activation**: ReLU</li></ul>
|
|**Layer 4:**<ul><li>**Fully Connected**: Input = 120. Output = 84.</li><li>**Dropout**</li><li>**Activation**: ReLU</li></ul>
|
|**Layer 5:**<ul><li>**Fully Connected (Logits)**: Input = 84. Output = 43.</li><li>**Softmax**: convert to probabilities</li></ul>
|

#### 4. Model Training

The code for model training follows immediately after the previous section.

The **softmax cross entropy** was the cost function used for the optimizer.

The **Adam optimizer** was used (lifted straight from the LeNet class exercise). It is a Stochastic Gradient Descent (SGD) optimizer with ability to schedule the learning rate adaptively based on the 1st & 2nd moments of the gradients.

Additionally, **mini-batching** was used.

Ten (10) Epochs of traning were run, with batch size of 128.

The **dropout keep probability** was set at 75% for the training. (No dropout, 100% keep, were used for model predictions.)

#### 5. Solution approach

The code for calculating the accuracy of the model is located in the code cells as above section of the Jupyter notebook.

My final model results were:
* training set accuracy of 98.8%
* validation set accuracy of 93.4%
* test set accuracy of 91.6%

As discussed earlier, the LeNet model was used as the starting point since it showed good performance in image identification during the class exercise. (ie: "lazy engineer" approach - build upon what already works! :D ) The LeNet model were slightly modified to accommondate the input / output data size.

he LeNet model by itself was having high-variance / over-fitting issue (good training performaced coupled with poor cross-validation performance), thus dropout layers were used to mitigate. This, coupled with feature normalization in the data pre-processing, were sufficient to achieve the above results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found the following collage of German traffic signs [online](https://s-media-cache-ak0.pinimg.com/originals/ce/55/f8/ce55f8319078dab5dbc37c51a77a837f.jpg), and extracted 30 signs from it.
![collage of German traffic signs][image3]



Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
