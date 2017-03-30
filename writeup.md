# Traffic Sign Recognition 

## Project Writeup

by Greg Yeutter

2017-03-30

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

[image1]: ./results/visualization.png "Visualization"
[image2]: ./results/process1.png "Preprocessing result 1"
[image3]: ./results/process2.png "Preprocessing result 2"
[image4]: ./results/process3.png "Preprocessing result 3"
[image5]: ./results/ext1.png "Traffic Sign 1"
[image6]: ./results/ext2.png "Traffic Sign 2"
[image7]: ./results/ext3.png "Traffic Sign 3"
[image8]: ./results/ext4.png "Traffic Sign 4"
[image9]: ./results/ext5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yeutterg/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the "Step 1" section of the Jupyter notebook. It is the first code cell of that section.

I used numpy and python functions to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) (3 for each RGB channel)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the rest of the "Step 1" section of the Jupyter notebook.  

The first exploration of the dataset lists the number of each class, the description of the trffic sign, and the count of that class in the dataset:

* Class0, Desc: Speed limit (20km/h), Count: 180
* Class1, Desc: Speed limit (30km/h), Count: 1980
* Class2, Desc: Speed limit (50km/h), Count: 2010
* Class3, Desc: Speed limit (60km/h), Count: 1260
* Class4, Desc: Speed limit (70km/h), Count: 1770
* Class5, Desc: Speed limit (80km/h), Count: 1650
* Class6, Desc: End of speed limit (80km/h), Count: 360
* Class7, Desc: Speed limit (100km/h), Count: 1290
* Class8, Desc: Speed limit (120km/h), Count: 1260
* Class9, Desc: No passing, Count: 1320
* Class10, Desc: No passing for vehicles over 3.5 metric tons, Count: 1800
* Class11, Desc: Right-of-way at the next intersection, Count: 1170
* Class12, Desc: Priority road, Count: 1890
* Class13, Desc: Yield, Count: 1920
* Class14, Desc: Stop, Count: 690
* Class15, Desc: No vehicles, Count: 540
* Class16, Desc: Vehicles over 3.5 metric tons prohibited, Count: 360
* Class17, Desc: No entry, Count: 990
* Class18, Desc: General caution, Count: 1080
* Class19, Desc: Dangerous curve to the left, Count: 180
* Class20, Desc: Dangerous curve to the right, Count: 300
* Class21, Desc: Double curve, Count: 270
* Class22, Desc: Bumpy road, Count: 330
* Class23, Desc: Slippery road, Count: 450
* Class24, Desc: Road narrows on the right, Count: 240
* Class25, Desc: Road work, Count: 1350
* Class26, Desc: Traffic signals, Count: 540
* Class27, Desc: Pedestrians, Count: 210
* Class28, Desc: Children crossing, Count: 480
* Class29, Desc: Bicycles crossing, Count: 240
* Class30, Desc: Beware of ice/snow, Count: 390
* Class31, Desc: Wild animals crossing, Count: 690
* Class32, Desc: End of all speed and passing limits, Count: 210
* Class33, Desc: Turn right ahead, Count: 599
* Class34, Desc: Turn left ahead, Count: 360
* Class35, Desc: Ahead only, Count: 1080
* Class36, Desc: Go straight or right, Count: 330
* Class37, Desc: Go straight or left, Count: 180
* Class38, Desc: Keep right, Count: 1860
* Class39, Desc: Keep left, Count: 270
* Class40, Desc: Roundabout mandatory, Count: 300
* Class41, Desc: End of no passing, Count: 210
* Class42, Desc: End of no passing by vehicles over 3.5 metric tons, Count: 210

It appeared that some classes had more items than others. This histogram shows the count of each class (class number on the x axis): 

![histogram][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the first code cell of "Step 2" in the Jupyter notebook.

Three techniques were used to preprocess images:
* Convert to grayscale using the cv2 library (flatten to 1 color channel)
* Normalize the image from [0, 255] to [-1, 1] using cv2
* Equialize the image histogram using the skimage library

Here are some examples of processed images:

![preprocessed image 1][image2]
![preprocessed image 2][image3]
![preprocessed image 3][image4]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The project data was pre-split into training, validation, and test sets. The training, validation, and test sets were all converted to grayscale, normalized, and equalized.

The training set contained 34,799 images. The validation set contained 4,410 images, and the test set contained 12,630 images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is located in the 3rd code cell of Step 2, under Model Architecture. The final model is a modified version of LeNet, that includes dropout on the third and fourth layers (out of five).

The final model consists of the following layers:

| Layer         		    | Description	        				    	| 
|:-------------------------:|:---------------------------------------------:| 
| 1a. Input         		| 32x32x1 Grayscale image   				    | 
| 1b. Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| 1c. RELU					|												|
| 1d. Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| 2a. Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| 2b. RELU          		|        									    |
| 2c. Max pooling			| 2x2 stride, valid padding, outputs 5x5x16     |
| 2d. Flatten       		| outputs 400								    |
| 3a. Fully connected		| outputs 120								    |
| 3b. RELU                  |                                               |
| 3c. Dropout               |                                               |
| 4a. Fully connected       | outputs 84                                    |
| 4b. RELU                  |                                               |
| 4c. Dropout               |                                               |
| 5a. Fully connected       | outputs 43                                    |
| 5b. Logits                |                                               | 

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

keep prob for training

The code for training the model is located in the fourth and fifth code cell of Step 2, under Train, Validate and Test the Model. Parameters for training are found under Model Architecture.

Some parameter tuning resulted in a 96.5% validation accuracy over 30 epochs. These were the parameters that led to this result:
* Batch size: 128
* Mu: 0
* Sigma: 0.01
* Dropout probability: 0.7
* Learning rate: 0.002

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the Train, Test, and Validate section of the notebook.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.5% (over 30 epochs) 
* test set accuracy of 92.6%

An iterative approach was chosen, modifying parameters of the default LeNet architecture.

* What was the first architecture that was tried and why was it chosen?

LeNet, as it was recommended for the project and is used in image classification tasks.

* What were some problems with the initial architecture?

While LeNet was functional, its validation accuracy was too low to meet project requirements.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In addition to parameter tuning, dropout was added at the end of the third and fourth layers. In general, this approach has the effect of preventing overfitting and enhancing the accuracy of the test and validation sets. The best validation accuracy was found with a dropout probability of 0.70. 

* Which parameters were tuned? How were they adjusted and why?

The number of epochs were increased, as the default of 10 epochs was not enough to truly evaluate model performance.

Sigma was adjusted, although the default probability of 0.01 ended up providing the best performance.

The growth rate was adjusted to 0.002, as higher values showed overfitting behavior.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolution layer produces an activation map of the input image, based on a defined quantity of filters. When combined with RELU, a threshold is defined on what features to keep or reject. Convolutional neural networks are designed to work in image applications.

Max pooling reduces the dimensionality of the input, so assumptions can be made about various regions of the image.

Fully connected layers are used to learn nonlinear combinations of feature outputs from previous layers.

Dropout prevents overfitting on various layers. Fully connected layers are especially prone to overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![speed limit 120][image5] ![stop][image6] ![keep right][image7] 
![roundabout][image8] ![end of speed limit 120][image9]

The first image might be difficult to classify because there are a number of speed limit signs that look similar, yet contain different values.

The second image might be difficult to classify due to the angle and the abnormal white text below "STOP."

The third and fourth images could be difficult to classify because a number of signs have a similar design (blue circle, white border, white symbol).

The fifth image was not actually represented in the dataset. A similar end of speed limit sign was represented, but at 80 km/h instead of 120 km/h.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in Step 3, under Predict the Sign Type for Each Image.

Here are the results of the prediction:

| Image			                | Prediction	        					    | Correct?       |
|:-----------------------------:|:---------------------------------------------:|:--------------:|
| Speed limit (120km/h)      	| Speed limit (120km/h)   						| Yes            |   
| Stop     			            | Speed limit (50km/h) 							| No             |   
| Keep right					| No entry										| No             |
| Roundabout mandatory	      	| End of no passing					 			| No             |
| End of speed limit (120km/h)	| End of speed limit (80km/h)      				| Yes (caveat)   |

Total correct: 2/5
Accuracy: 40%

For the end of speed limit sign, although the exact sign was not represented in the dataset, the closest (and expected) match was identified.

An accuracy of 40% does not reflect the 90+% accuracies of the validation and test sets. An improved model may be able to increase this accuracy, as well as a more diverse training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Step 3, under Output Top 5 Softmax Probabilities.

For the first image, the model is very sure that this is a speed limit of 120 km/h sign (probability of 0.97), which is correct. The next 4 predictions were also speed limit signs. The top five soft max probabilities were:

| Probability         	| Prediction	             					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Speed limit (120km/h)   						| 
| .02     				| Speed limit (20km/h) 							|
| .00					| Speed limit (30km/h)							|
| .00	      			| Speed limit (100km/h)					 		|
| .00				    | Speed limit (70km/h)      					|


For the second image, the correct prediction (stop sign) is not even listed. However, the top 5 results have similar, albeit inverted, color profiles. 

| Probability         	| Prediction	        				    	| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Speed limit (50km/h)   						| 
| .02     				| Wild animals crossing 						|
| .01					| peed limit (80km/h)							|
| .00	      			| Speed limit (30km/h)				 		    |
| .00				    | Speed limit (60km/h)      					|


For the third image, the model is almost entirely positive that this is a no entry sign, which is incorrect.

| Probability         	| Prediction	        				    	| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry  					            	| 
| .00     				| Stop 						                    |
| .00					| Go straight or left							|
| .00	      			| Speed limit (20km/h)				 		    |
| .00				    | Roundabout mandatory      					|


For the fourth image, the correct prediction did not appear in the top five softmax probabilities. The model was very unsure, with the highest probability at 51%. Interestingly, the top prediction (End of no passing) ranked fifth in the softmax probabilities.

| Probability         	| Prediction	        				    	| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Keep right  					            	| 
| .18     				| No entry						                |
| .15					| Speed limit (20km/h)							|
| .07	      			| Dangerous curve to the right				    |
| .06				    | End of no passing      				        |

For the fifth image, even though the identified image is does not technically match (80 vs. 120 km/h), it is the closest image in the dataset. Both the prediction and softmax probability agree, with softmax giving the 80 km/h sign 83%.

| Probability         	| Prediction	        				    	        | 
|:---------------------:|:-----------------------------------------------------:| 
| .83         			| End of speed limit (80km/h)  					        | 
| .15     				| End of all speed and passing limits                   |
| .01					| End of no passing by vehicles over 3.5 metric tons	|
| .00	      			| End of no passin				                        |
| .00				    | Speed limit (30km/h)      				            |