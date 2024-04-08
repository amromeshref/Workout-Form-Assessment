from src.exception import CustomException
from src.logger import logging
import os
#from tensorflow.keras.models import Sequential
import sys
import numpy as np
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, exercise_name: str, evaluation_type: str):
        logging.info("Training Process Started")
        self.exercise_name = exercise_name
        self.evaluation_type = evaluation_type

    def get_data(self):
        """
        This function will prepare the data for training and testing
        Input: None
        Output: training_data_criteria1, training_data_criteria2, training_data_criteria3, training_data_criteria4
                The training and testing data for each criteria
        """
        try:
            dir_path = os.path.normpath(os.path.join(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/processed")))
            dir_path += "/" + self.exercise_name+"/" + self.evaluation_type
            data_criteria1_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_1.npy"
            data_criteria1 = self.split_data_per_criteria(
                data_criteria1_path)
            data_criteria2_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_2.npy"
            data_criteria2 = self.split_data_per_criteria(
                data_criteria2_path)
            data_criteria3_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_3.npy"
            data_criteria3 = self.split_data_per_criteria(
                data_criteria3_path)
            data_criteria4_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_4.npy"
            data_criteria4 = self.split_data_per_criteria(
                data_criteria4_path)
            return data_criteria1, data_criteria2, data_criteria3, data_criteria4
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def split_data_per_criteria(self, training_data_path: str):
        """
        Split the data into training and testing data for each criteria
        Input: training_data_path
        Output: a list contains: X_train, y_train, X_test, y_test
        """
        try:
            training_data = np.load(training_data_path, allow_pickle=True)
            X = np.vstack(training_data[:, 0])
            y = np.vstack(training_data[:, 1])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            return [X_train, y_train, X_test, y_test]
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def train(self):
        pass
