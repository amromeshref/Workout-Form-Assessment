# Data Preprocessing

This directory contains scripts and modules related to the preprocessing of data for model training or analysis. 
Data preprocessing is an essential step in the machine learning pipeline, involving tasks such as data cleaning, transformation, and 
feature engineering to prepare the data for further analysis or modeling.

## Directory Structure

- `cycles_divider.py`: Module for dividing video data into cycles based on exercise movements.
- `data_transformation.py`: Module for transforming video data into the required format for model training.
- `external_to_interim_transformation.py`: Script for transforming external data into interim data.
- `interim_to_processed_transformation.py`: Script for transforming interim data into processed data.

## Scripts

### 1. `cycles_divider.py`
- `cycles_divider.py` provides functions for dividing video data into cycles, which can be used to segment exercises into meaningful
  units for analysis or modeling.

### 2. Data Transformation
- `data_transformation.py` contains the `DataTransformer` class, which transforms video data into the required format for model training.
  It includes methods for extracting poses and angles from video frames, as well as preparing training data.

### 3. External to Interim Transformation
- `external_to_interim_transformation.py` is a script that transforms external data into interim data. The transformation primarily involves
  dividing the videos into cycles, where each cycle represents a repetition of the exercise movement.

### 4. Interim to Processed Transformation
- `interim_to_processed_transformation.py` is a script that transforms interim data into processed data. It performs the crucial
  task of converting the cycles of the exercises present in the interim data into sequential data, which is suitable for training the models.


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
