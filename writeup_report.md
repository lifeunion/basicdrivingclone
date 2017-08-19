**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-model.png "Model Visualization"
[image2]: ./examples/center-lane.png "Center Lane Driving"

## Rubric Points
---

####1. Submission includes all required files and can be used to run the simulator in autonomous mode.

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

## Model Architecture Design

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It is chosen because I just want to stand on their sholder as the giant of the self driving car tests so far. They used this model to train their cars.

This deep convolutional network has been shown to work really well for regression purposes. On top of that, I try to prevent overfitting by adding dropout layer. I also added pre-process layer to feed cropped images into it.

I've added the following adjustments to the model:
- Lambda layer to normalize and centre-mean the data,
- Cropping2D to lessen storage and processing space/time
- Dropout layer to prevent overfitting as mentioned above,

In the end, the model looks like as follows:

- Image normalization
- Cropping
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Drop out (0.5)
- Fully connected: neurons: 1164, activation: RELU
- Fully connected: neurons:  100, activation: RELU
- Fully connected: neurons:   50, activation: RELU
- Fully connected: neurons:   10, activation: RELU
- Fully connected: neurons:    1 (output)

The model's goal is to mimic my driving behavior, specifically steering angles when I perform the driving by keyboard as the data. Overall, the model is very functional to clone that behavior.

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 160, 320,3)|0       |lambda_input_1   |
|cropping2d_1                    |(None, 80, 320, 3)|0       |lambda_1         |
|convolution2d_1 (Convolution2D) |(None, 38, 158,24)|1824    |cropping2d_1     |
|convolution2d_2 (Convolution2D) |(None, 17, 77, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 7, 37, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 5, 35, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 3, 33, 64) |36928   |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 3, 33, 64) |0       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 6636)      |0       |dropout_1        |
|dense_1 (Dense)                 |(None, 1164)      |7376268 |flatten_1        |
|dense_2 (Dense)                 |(None, 100)       |116500  |dense_1          |
|dense_3 (Dense)                 |(None, 50)        |5050    |dense_2          |
|dense_4 (Dense)                 |(None, 10)        |510     |dense_3          |
|dense_5 (Dense)                 |(None, 1)         |11      |dense_4          |
|                                |**Total params**  |252219  |                 |

![alt text][image1]

####2. Attempts to reduce overfitting in the model

Dropout layer is added after convolutions in order to reduce overfitting (model.py lines 73).
The data set was also always shuffled during training process with 20% training data used for validation.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was ran through autonomous mode on the simulator. It is verified by seeeing through that the vehicle stays within the track.

####3. Model parameter tuning

The model used an adam optimizer so that manually training the learning rate wasn't necessary. (model.py line 87).

####4. Appropriate training data

Training data was intentionally created with 3 main emphasis as the nature of the drive.
- to drive on the center lane as much as possible
- big angle steering on the curves
- staying within tracks
Of course, all of them are done with much recovering from the sides.

![alt text][image2]

The final step was to run the simulator to see how well the car was driving around track one. The first curve wasnt passed in the first try. I added 3 more data run by jerk-steering big angle on the curves. The second and the third curves were also passed after addding that kind of data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
