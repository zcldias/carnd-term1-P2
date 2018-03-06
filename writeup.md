# **Traffic Sign Recognition**



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

#### Here I use a class called data to manage the dataset, I did the preprocessing in this class. Besides, the data augmentation is also performed in this class.

* First, I load the train, val, test data by function `load_data()`
* Then check the number of the data and label
* normalize the grayscale the images by function `normal_grayscale()` (Here I use the skimage to do the image process)
* Augment training data by using function `expend_training_data()`
* I wirte a function called `next_batch()` to feed random minibatch to the model

#### After data augmentation, the information of the dataset is printed by the function `print_data_info()`:

> * Number of training examples = **173995**
> * Number of validation examples = **4410**
> * Number of testing examples = **12630**
> * Image data shape = **(32, 32)**
> * Number of classes = **43**

#### 2. Include an exploratory visualization of the dataset.

## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The image prepossing is done in the `data` class, which include the normalization, grayscale and data agumentation.

#### Data augmentation

The number of train-data is increased to 5 times by means of

* Random rotation : each image is rotated by random degree in ranging [-15°, +15°].
* Random shift : each image is randomly shifted by a value ranging [-2pix, +2pix] at both axises.
* Zero-centered normalization : a pixel value is subtracted by (PIXEL_DEPTH/2) and divided by PIXEL_DEPTH.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In this stage, I write a class called LeNet. I have not change the structure of the LeNet model. The kernal size, input size, output size, hidden neurons are all the same with lectures. I add the following code:
* First, I used the namespace and variable space to manage the paramenters instead of the dictionary
* I write some helper function like `conv2d()`, `maxpool2d()` and `fc_layer()`
* I write some information to summary so we can use **tensorboard** to visualize
* The model is saved to a checkpoint file by using saver
* Here I use the Adam Optimizer to train the model
* the learning rate is 0.001


### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

#### Training stage
* I train the model for 20 epoch
* The batch size I used is 128
* the model is saved to the 'model_save_dir'
* the init learning rate is set at 0.001
* I use the Adam Optimizer to train the model

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97.07%
* test set accuracy of 95.03%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  I just used the conventional LeNet model and have not change it at all.

* What were some problems with the initial architecture?

  It is easy to be overfit.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  To overcome the overfitting problem, I add two dropout layer and set the keep prob at 0.5. Then my model has a valid accuracy more than 0.93.

* Which parameters were tuned? How were they adjusted and why?

  I add dropout and use data agumentation

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  The dropout.

If a well known architecture was chosen:
* What architecture was chosen?

  Traditional LeNet

* Why did you believe it would be relevant to the traffic sign application?

  It is very similar to the MNIST dataset

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  the valid and test accuracy is very close and higher than 0.93


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model work fine when the sign is clear. But when the sign is not in the train dataset, is still hard to identify the sign precisily.

The prediction accuracy is 60%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 is calculate in the model:

```
[[  1.00000000e+00   1.06447605e-19   4.05146957e-20   4.74103778e-21
    1.87866530e-23]
 [  1.00000000e+00   7.37162864e-15   1.63145493e-21   6.15492629e-23
    5.56724132e-23]
 [  1.00000000e+00   4.69189401e-32   8.88218357e-38   0.00000000e+00
    0.00000000e+00]
 [  1.00000000e+00   1.64398304e-11   2.35198119e-13   4.66793957e-23
    3.07634107e-23]
 [  3.83373827e-01   1.56300962e-01   1.11141562e-01   8.24019611e-02
    6.80121854e-02]
 [  1.00000000e+00   4.58765070e-09   1.50478519e-10   3.26221689e-12
    1.66451293e-12]
 [  1.00000000e+00   1.31442593e-10   5.46305645e-17   2.76542291e-21
    2.44920667e-21]
 [  7.62751281e-01   8.67487341e-02   6.40631244e-02   3.11125182e-02
    2.25958340e-02]
 [  9.86572623e-01   1.06138643e-02   2.18002498e-03   2.88457784e-04
    2.83240195e-04]
 [  9.53132510e-01   1.69339385e-02   6.44845702e-03   6.18995447e-03
    5.28505724e-03]]
[[38 13  5  2 34]
 [25 22 20 29 13]
 [13 15 12  0  1]
 [ 1  2  0  5  4]
 [33 36 18 14  1]
 [23 29 31 20 19]
 [17 14 33 34  9]
 [34 14 36 35 25]
 [11 27 30 23 28]
 [36 18 20 25 35]]
 ```

