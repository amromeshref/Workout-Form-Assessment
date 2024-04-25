# Visualization Directory

This directory contains scripts for visualizing various aspects of the project. The visualization scripts are designed to provide visual 
feedback and analysis for different stages of the project. These scripts utilize computer vision techniques to process video input 
and generate visual representations of key metrics, such as pose estimation, cycles division, and feedback visualization.


## Scripts

### 1. `visualize_cycles_divider.py`

This script visualizes the process of dividing exercise cycles from a video source. It identifies key poses and angles to 
segment the video into distinct exercise cycles.

#### Usage:
- If you want the input source to be a locally saved video file:
  `python visualize_cycles_divider.py <exercise_name> video --path <video_path>`
  - `exercise_name`: Name of the exercise: `bicep` or `lateral_raise`.
  - `<video_path>`: Path to the video file.
- If you prefer to use a live webcam feed as the input source:
  `python visualize_cycles_divider.py <exercise_name> webcam`
  - `exercise_name`: Name of the exercise: `bicep` or `lateral_raise`.
- The cycles after running this script are saved at `results/cycles_divider`.


### 2. `visualize_feedback.py`

This script visualizes feedback based on exercise performance. It uses trained models to evaluate exercise criteria and provides 
feedback on each cycle of the exercise.

#### Usage:
- If you want the input source to be a locally saved video file:
  `python visualize_feedback.py <exercise_name> <evaluation_type> video --path <video_path>`
  - `exercise_name`: Name of the exercise: `bicep` or `lateral_raise`.
  - `evaluation_type`: Type of evaluation criteria: `poses` or `angles`.
  - `<video_path>`: Path to the video file.
- If you prefer to use a live webcam feed as the input source:
  `python visualize_feedback.py <exercise_name> <evaluation_type> webcam`
  - `exercise_name`: Name of the exercise: `bicep` or `lateral_raise`.
  - `evaluation_type`: Type of evaluation criteria: `poses` or `angles`.
- The results after running this script is saved at `results/feedback`.


### 3. visualize_pose_estimation.py

This script visualizes the process of pose estimation from a video source. It uses the Mediapipe Pose Estimation model to detect and 
visualize human poses in real-time or from a video file.

#### Usage:
- If you want the input source to be a locally saved video file:
  `python visualize_pose_estimation.py video --path <video_path>`
  - `<video_path>`: Path to the video file.
- If you prefer to use a live webcam feed as the input source:
  `python visualize_pose_estimation.py webcam`
