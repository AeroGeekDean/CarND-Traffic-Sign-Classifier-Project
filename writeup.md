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
[image101]: ./test_data_from_web/sample_01.jpg "Traffic Sign 1"
[image102]: ./test_data_from_web/sample_02.jpg "Traffic Sign 2"
[image103]: ./test_data_from_web/sample_03.jpg "Traffic Sign 3"
[image104]: ./test_data_from_web/sample_04.jpg "Traffic Sign 4"
[image105]: ./test_data_from_web/sample_05.jpg "Traffic Sign 5"
[image106]: ./test_data_from_web/sample_06.jpg "Traffic Sign 6"
[image107]: ./test_data_from_web/sample_07.jpg "Traffic Sign 7"
[image108]: ./test_data_from_web/sample_08.jpg "Traffic Sign 8"
[image109]: ./test_data_from_web/sample_09.jpg "Traffic Sign 9"
[image110]: ./test_data_from_web/sample_10.jpg "Traffic Sign 10"
[image111]: ./test_data_from_web/sample_11.jpg "Traffic Sign 11"
[image112]: ./test_data_from_web/sample_12.jpg "Traffic Sign 12"
[image113]: ./test_data_from_web/sample_13.jpg "Traffic Sign 13"
[image114]: ./test_data_from_web/sample_14.jpg "Traffic Sign 14"
[image115]: ./test_data_from_web/sample_15.jpg "Traffic Sign 15"
[image116]: ./test_data_from_web/sample_16.jpg "Traffic Sign 16"
[image117]: ./test_data_from_web/sample_17.jpg "Traffic Sign 17"
[image118]: ./test_data_from_web/sample_18.jpg "Traffic Sign 18"
[image119]: ./test_data_from_web/sample_19.jpg "Traffic Sign 19"
[image120]: ./test_data_from_web/sample_20.jpg "Traffic Sign 20"
[image121]: ./test_data_from_web/sample_21.jpg "Traffic Sign 21"
[image122]: ./test_data_from_web/sample_22.jpg "Traffic Sign 22"
[image123]: ./test_data_from_web/sample_23.jpg "Traffic Sign 23"
[image124]: ./test_data_from_web/sample_24.jpg "Traffic Sign 24"
[image125]: ./test_data_from_web/sample_25.jpg "Traffic Sign 25"
[image126]: ./test_data_from_web/sample_26.jpg "Traffic Sign 26"
[image127]: ./test_data_from_web/sample_27.jpg "Traffic Sign 27"
[image128]: ./test_data_from_web/sample_28.jpg "Traffic Sign 28"
[image129]: ./test_data_from_web/sample_29.jpg "Traffic Sign 29"
[image130]: ./test_data_from_web/sample_30.jpg "Traffic Sign 30"
[image4]: ./writeup_images/web_img_performance.jpg "New image prediction performance"

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

While looking at some of the training images during the Exploratory Data visualization stage earlier, it was noticed that often the images are dark and have poor contrast. Thus contrast normalization were performed on each image individually. This was accomplished by scaling each images contrast to be between the range of [0, 255], utilizing `'OpenCV2.normalize()'` function.

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

Ten (10) Epochs of training were run, with batch size of 128.

The **dropout keep probability** was set at 75% for the training. (No dropout, 100% keep, were used for model predictions.)

#### 5. Solution approach

The code for calculating the accuracy of the model is located in the code cells as above section of the Jupyter notebook.

My final model results were:
* training set accuracy of 98.8%
* validation set accuracy of 93.4%
* test set accuracy of 91.6%

As discussed earlier, the LeNet model was used as the starting point since it showed good performance in image identification during the class exercise. (ie: "lazy engineer" approach - build upon what already works! :D ) The LeNet model were slightly modified to accommodate the input / output data size.

he LeNet model by itself was having high-variance / over-fitting issue (good training performance coupled with poor cross-validation performance), thus dropout layers were used to mitigate. This, coupled with feature normalization in the data pre-processing, were sufficient to achieve the above results.

### Test a Model on New Images

The code for this portion is contained under the **"Step 3: Test a Model on New Images"** section of the Jupyter notebook.

#### 1. New Images From the Web

I found the following collage of German traffic signs [online](https://s-media-cache-ak0.pinimg.com/originals/ce/55/f8/ce55f8319078dab5dbc37c51a77a837f.jpg), and extracted 30 signs from it.
![collage of German traffic signs][image3]

Below are the 30 extracted examples:

|1![alt text][image101]| 2![alt text][image102]| 3![alt text][image103]| 4![alt text][image104]| 5![alt text][image105]| 6![alt text][image106]| 7![alt text][image107]| 8![alt text][image108]| 9![alt text][image109]| 10![alt text][image110]|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|11![alt text][image111]| 12![alt text][image112]| 13![alt text][image113]| 14![alt text][image114]| 15![alt text][image115]| 16![alt text][image116]| 17![alt text][image117]| 18![alt text][image118]| 19![alt text][image119]| 20![alt text][image120]|
|21![alt text][image121]| 22![alt text][image122]| 23![alt text][image123]| 24![alt text][image124]| 25![alt text][image125]| 26![alt text][image126]| 27![alt text][image127]| 28![alt text][image128]| 29![alt text][image129]| 30![alt text][image130]|

Some of the image classes are **not** in the training set, and were **intentionally** chosen because of my curiosity to see how the model will perform on them. (Example: 3, 8, 10, 12, 21, 22, 23, 30)

#### 2. Predict the Sign Type for Each Image

Below is graphical results visualizing the prediction performance on each of these 30 images.

Explanation on each element of the visualization are as follow:
- Left column: sample image shown
- Center column: plot of probabilities on the classification of the image
  - X-axis = the 43 possible Classifications (0-42)
  - Y-axis = softmax probability that the image is of that class
  - The sum (area under the curve) of all the probabilities should = 1.0
- Right column: listing of top 3 probable predictions
  - The probability [%] is provided
  - The classification label number and its associated description are written out
  - Color codes:
    - Green = Correct prediction
    - Red = Incorrect prediction
    - Blue = Image type is **NOT** a part of the original 43 classifications that the model was trained on.
      - Yes, it was unfair to the model! :)
  - **Bold** text = Prediction confidence > 90% probability (a number I arbitrary chose)

![New images prediction performance][image4]

Of the 30 sample images, 8 were unfair for the model to try to predict (the blue coded labels). Of the remaining 22 images, the model predicted 19 correctly, and 3 incorrectly. **This gives an accuracy of 19/22 = 86.4%**

>**Note**
>
>If I had more time to really analyze it, we could build a Confusion Matrix *for each class* in order to come up with classification-specific precision and recall metrics.
>
>Perhaps tune each classification's probability threshold based on its ROC (Receiver Operating Characteristic) curve to optimize its metrics.
>
>But would threshold tuning cause cross-classification issues, in a multi-classification situation? Hmmm.... will have to dig into this more in the future. I'm running out of time to meet my Term 1 deadline, EVEN WITH the automatic 4-weeks extension (after missing the original term1 deadline)!
>:(

**Observations**:

As can be seen from the classification probability ("confidence") plots, the model:
1. Got 'lucky' on some images:
>low confidence but got it right! (ex: image #17 "30 km/h Speed Limit", denoted by non-bold green text)
2. 'Almost, but not quite' on some images:
>correct prediction was the 2nd highest probability, just slightly under the top choice (ex: image #24 "No Passing")
3. 'It wasn't fair!' complains the model...
>Image #28 "Priority road" sign was combined with another sign, thus confused the model. The model wasn't trained on distinguishing multiple signs within an image.
4. It's also interesting to observe how the model attempts to predict on images types that it was not trained on. (Blue color code)
