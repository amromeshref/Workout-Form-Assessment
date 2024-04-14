import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

import numpy as np
from tensorflow.nn import sigmoid
from tensorflow.keras.models import load_model
from src.logger import logging
from src.exception import CustomException



class ModelPredictor:
    def __init__(self, exercise_name: str, evaluation_type: str):
        # Initialize the ModelPredictor class with exercise name and evaluation type
        logging.info("Prediction Started for Exercise: " +
                     exercise_name+" Evaluation Type: "+evaluation_type)
        # Define paths to the trained model files for each criteria
        self.model_path_criteria1 = os.path.join(
            REPO_DIR_PATH, "models", "best", exercise_name, evaluation_type, exercise_name+"_criteria1_"+evaluation_type+"_model.h5")
        self.model_path_criteria2 = os.path.join(
            REPO_DIR_PATH, "models", "best", exercise_name, evaluation_type, exercise_name+"_criteria2_"+evaluation_type+"_model.h5")
        self.model_path_criteria3 = os.path.join(
            REPO_DIR_PATH, "models", "best", exercise_name, evaluation_type, exercise_name+"_criteria3_"+evaluation_type+"_model.h5")
        # Load the trained models
        self.model_criteria1 = load_model(self.model_path_criteria1)
        self.model_criteria2 = load_model(self.model_path_criteria2)
        self.model_criteria3 = load_model(self.model_path_criteria3)

    def predict_criteria1(self, input_sequential_data: np.ndarray):
        """
        Predicts the criteria 1 for the given sequential data
        Args:
            input_sequential_data (np.ndarray): Sequential data to predict the criteria 1
        Returns:
            float: Prediction for the criteria 1        
        """
        try:
            # Check if input data is 2D, if yes, reshape it to 3D
            if len(input_sequential_data.shape) == 2:
                input_sequential_data = input_sequential_data.reshape(
                    (1, input_sequential_data.shape[0], input_sequential_data.shape[1]))
            # Make predictions using the model for criteria 1
            predictions = self.model_criteria1.predict(input_sequential_data)
            # Apply sigmoid activation function
            predictions = sigmoid(predictions)
            logging.info("Prediction Completed for Criteria 1")
            return predictions[0][0].numpy()
        except Exception as e:
            logging.error("Prediction Failed for Criteria 1 due to: "+str(e))
            raise CustomException("Prediction Failed for Criteria 1")

    def predict_criteria2(self, input_sequential_data: np.ndarray):
        """
        Predicts the criteria 2 for the given sequential data
        Args:
            input_sequential_data (np.ndarray): Sequential data to predict the criteria 2
        Returns:
            float: Prediction for the criteria 2        
        """
        try:
            # Check if input data is 2D, if yes, reshape it to 3D
            if len(input_sequential_data.shape) == 2:
                input_sequential_data = input_sequential_data.reshape(
                    (1, input_sequential_data.shape[0], input_sequential_data.shape[1]))
            # Make predictions using the model for criteria 2
            predictions = self.model_criteria2.predict(input_sequential_data)
            # Apply sigmoid activation function
            predictions = sigmoid(predictions)
            logging.info("Prediction Completed for Criteria 2")
            return predictions[0][0].numpy()
        except Exception as e:
            logging.error("Prediction Failed for Criteria 2 due to: "+str(e))
            raise CustomException("Prediction Failed for Criteria 2")

    def predict_criteria3(self, input_sequential_data: np.ndarray):
        """
        Predicts the criteria 3 for the given sequential data
        Args:
            input_sequential_data (np.ndarray): Sequential data to predict the criteria 3
        Returns:
            float: Prediction for the criteria 3        
        """
        try:
            # Check if input data is 2D, if yes, reshape it to 3D
            if len(input_sequential_data.shape) == 2:
                input_sequential_data = input_sequential_data.reshape(
                    (1, input_sequential_data.shape[0], input_sequential_data.shape[1]))
            # Make predictions using the model for criteria 3
            predictions = self.model_criteria3.predict(input_sequential_data)
            # Apply sigmoid activation function
            predictions = sigmoid(predictions)
            logging.info("Prediction Completed for Criteria 3")
            return predictions[0][0].numpy()
        except Exception as e:
            logging.error("Prediction Failed for Criteria 3 due to: "+str(e))
            raise CustomException("Prediction Failed for Criteria 3")
