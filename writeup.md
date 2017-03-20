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
[image3]: ./examples/random_noise.jpg "Random Noise"
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

#### Data Pre-processing

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

#### Validation Strategy

The dataset provided has already been split out for training/cross-validation/testing. The split is about:

| Data Set | % |
|:-----:|:-----:|
| Training | 67.1 |
| Cross-Validation | 8.5 |
| Testing | 24.4 |

Another validation strategy is to utilize the [K-Fold Cross-Validation method](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation), unfortunately I ran out of time thus did not implement for this project... :\

#### Model Architecture

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **Model Architecture** section of the Jupyter notebook. 

I started with the LeNet architecture provided straight from the lecture, adjusting for the input and output sizes. This gave unacceptable performance over-fitting results, thus a **dropout layer** was added before each activation.

My final model consisted of the following:

| Model Layers |
|:-----|
| **Layer 1:**
| >**Convolution** Input = 32x32x3. Filter = (5x5x3)x6. Output = 28x28x6.
|
|>**Dropout**
|
|>**Activation** ReLU
|
|>**Pooling** Input = 28x28x6. Filter = 2x2. Output = 14x14x6.
|
|**Layer 2:**
|>**Convolution** Input = 14x14x6. Filter = (5x5x6)x16. Output = 10x10x16.
|
|>**Dropout**
|
|>**Activation** ReLU
|
|>**Pooling** Input = 10x10x16. Filter = 2x2. Output = 5x5x16.
|
|**Flatten.** Input = 5x5x16. Output = 400.
|
|**Layer 3:**
|>**Fully Connected** Input = 400. Output = 120.
|
|>**Dropout**
|
|>**Activation** ReLU
|
|**Layer 4:**
|>**Fully Connected** Input = 120. Output = 84.
|
|>**Dropout**
|
|>**Activation** ReLU
|
|**Layer 5:**
|>**Fully Connected (Logits)** Input = 84. Output = 43.
|


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

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
