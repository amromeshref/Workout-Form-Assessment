# Data Preprocessing

Welcome to the Data Preprocessing directory! This directory houses scripts and modules dedicated to the preprocessing of raw exercise data.
The scripts contained within this directory cover a range of preprocessing tasks, including transforming external data into interim data, dividing videos into cycles, and preparing training data for machine learning models.


## Scripts

### 1. `cycles_divider.py`
- It provides functions for dividing video data into cycles, which can be used to segment exercises into meaningful
  units for analysis or modeling.
- To retrieve the results from cycles divider, navigate to the `src/visualization` directory and execute the script
  `visualize_cycles_divider.py`.

### 2. `data_transformation.py`
- It contains the `DataTransformer` class, which transforms video data into the required format for model training.
  It includes methods for extracting poses and angles from video frames, as well as preparing training data.
- To run the script:
  
  ```
  python data_transformation.py <exercise_name> <evaluation_type>
  ```
  
  - `<exercise_name>` : Name of the exercise: `bicep` or `lateral_raise`.
  - `<evaluation_type>` :Evaluation type used for representing cycles as sequential data: `poses` or `angles`.
  - The sequential data for the chosen exercise and chosen evaluation type will be saved at `data/processed/<exercise_name>/<evaluation_type>`
  
### 3. `external_to_interim_transformation.py`
- It transforms external data into interim data. The transformation primarily involves
  dividing the videos into cycles, where each cycle represents a repetition of the exercise movement.
- To run the script:
  
  ```
  python external_to_interim_transformation.py
  ```
  - Videos in `data/external` will be divided into cycles and saved at `data/interim`.

### 4. `interim_to_processed_transformation.py`
- `It transforms interim data into processed data. It performs the crucial
  task of converting the cycles of the exercises present in the interim data into sequential data, which is suitable for training the models.
- To run the script:
  
  ```
  python interim_to_processed_transformation.py
  ```
  - Cycles in `data/interim` will be converted into sequential data (both poses and angles) and saved at `data/processed`.


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
