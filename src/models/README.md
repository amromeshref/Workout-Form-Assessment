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
- To obtain predictions from the models, navigate to the `src/visualization` directory and run the script `visualize_feedback.py`.

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

