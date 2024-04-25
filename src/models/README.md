# Exercise Models

Welcome to the Exercise Models directory! This directory contains scripts for training models and performing predictions on exercise data. 
The models are designed to analyze exercise videos and predict various evaluation criteria based on the movements observed.
The scripts provided in this directory enable users to train custom machine learning models tailored to specific exercises and 
evaluation criteria, as well as to make predictions using pre-trained models.

## Scripts

### 1. `predict_model.py`
- It is used for predicting evaluation criteria based on input sequential data using pre-trained models. It takes sequential data representing
  exercise cycles as input and predicts the probability for each evaluation criteria. When the probability is high,
  it indicates that the corresponding criteria is met.

### 2. `train_model.py`
- It is used for training the models using prepared training data.
- The results after the training is saved at `models/other/<exercise_name>/<evaluation_type>/<current_time>`
  - `<exercise_name>` : `bicep` or `lateral_raise`
  - `<evaluation_type>` : `poses` or `angles`
  - `<current_time>` : The time when you run the srcript
- To run the script:

  `python train_model.py <exercise_name> <evaluation_type>`

  - `<exercise_name>` : Name of the exercise: `bicep` or `lateral_raise`
  - `<evaluation_type>` :Evaluation type used for representing cycles as sequential data: `poses` or `angles`


## How Cycles Divider Works

The Cycles Divider module functions by analyzing video data of exercises and identifying repeated movements or cycles within the exercise. 
For the bicep exercise, it divides cycles based on the change in angle between the shoulder, elbow, and wrist. For the lateral raise exercise, 
it focuses on the angle between the hip, shoulder, and elbow.

1. **Angle Extraction and Analysis**

The module extracts angles over frames from the video of the exercise. It calculates these angles using key landmarks detected 
by pose estimation model, MediaPipe. The angles extracted typically exhibit a periodic pattern resembling a sine wave when 
plotted against the frame number. Each complete cycle of this wave represents one repetition of the exercise.

2. **Noise Reduction Filters**

To enhance the accuracy of cycle detection and reduce noise in the angle data, the module applies a combination of filters. 
This combination helps smooth out fluctuations caused by factors such as camera movement, lighting variations, or small variations in movement speed.
The primary noise reduction function applies both a Butterworth filter and a median filter to the angle data. This combination effectively reduces 
noise while preserving the essential features of the angle graph, leading to more accurate cycle detection and segmentation.

3. **Cycle Division**

To divide cycles, the module observes the change in angles over consecutive frames. It identifies periods of increasing and decreasing angles, 
signifying the completion of one cycle. These periods correspond to the peaks and troughs of the sine wave-like angle graph. By detecting these 
changes, the module segments the video data into individual cycles, enabling finer-grained analysis and modeling of exercise movements.


## Add later the representation techniques of the sequential data: poses/angles
