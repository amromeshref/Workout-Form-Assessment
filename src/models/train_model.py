import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization, GRU
from tensorflow.keras.models import Sequential
from src.logger import logging
from src.exception import CustomException


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
            # Define the directory path
            dir_path = os.path.normpath(os.path.join(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/processed")))
            dir_path += "/" + self.exercise_name+"/" + self.evaluation_type

            # Load data for criteria1 and split the data into training and testing data
            data_criteria1_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_1.npy"
            data_criteria1 = self.split_data_per_criteria(
                data_criteria1_path)
            
            # Load data for criteria2 and split the data into training and testing data
            data_criteria2_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_2.npy"
            data_criteria2 = self.split_data_per_criteria(
                data_criteria2_path)
            
            # Load data for criteria3 and split the data into training and testing data
            data_criteria3_path = dir_path+"/training_data_" + \
                self.exercise_name+"_"+self.evaluation_type+"_criteria_3.npy"
            data_criteria3 = self.split_data_per_criteria(
                data_criteria3_path)
            
            # Load data for criteria4 and split the data into training and testing data
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
            # Load training data
            training_data = np.load(training_data_path, allow_pickle=True)
            X = np.vstack(training_data[:, 0])
            y = np.vstack(training_data[:, 1])
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            return [X_train, y_train, X_test, y_test]
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)


    def create_model(self, input_shape: tuple):
        """
        This function will create the model for training
        Input: input_shape
        Output: model
        """
        model = Sequential(
            [
                GRU(256, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                GRU(128, return_sequences=True),
                BatchNormalization(),
                GRU(64, return_sequences=True),
                BatchNormalization(),
                GRU(32),
                BatchNormalization(),
                Dense(1, activation="linear")
            ]
        )
        return model

    def train(self):
        """
        This function will train the model for each criteria and save the models
        """
        # Get data for training
        data_criteria1, data_criteria2, data_criteria3, _ = self.get_data()
        X_train_criteria1, y_train_criteria1, _, _ = data_criteria1
        X_train_criteria2, y_train_criteria2, _, _ = data_criteria2
        X_train_criteria3, y_train_criteria3, _, _ = data_criteria3

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        # Create directory path for saving models
        dir_path = os.path.normpath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "other",
                                                                self.exercise_name, self.evaluation_type)))
        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Train and save model for criteria 1
        input_shape = (X_train_criteria1.shape[1], X_train_criteria1.shape[2])
        # Create the model
        model_criteria1 = self.create_model(input_shape)
        # Compile the model
        model_criteria1.compile(metrics=['accuracy'],
                                optimizer=Adam(learning_rate=0.0001),
                                loss=BinaryCrossentropy(from_logits=True))
        logging.info("Training Model for Criteria 1 Started")
        # Train the model
        model_criteria1.fit(X_train_criteria1,
                            y_train_criteria1, epochs=50, batch_size=32)
        logging.info("Training Model for Criteria 1 Completed")
        # Save the model
        saved_path = os.path.normpath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "other",
                                                                self.exercise_name, self.evaluation_type, current_time+"_"+self.exercise_name+"_criteria1_"+self.evaluation_type+"_model.h5")))
        model_criteria1.save(saved_path)
        logging.info("Model for Criteria 1 Saved at "+saved_path)

        # Train and save model for criteria 2
        input_shape = (X_train_criteria2.shape[1], X_train_criteria2.shape[2])
        # Create the model
        model_criteria2 = self.create_model(input_shape)
        # Compile the model
        model_criteria2.compile(metrics=['accuracy'],
                                optimizer=Adam(learning_rate=0.0001),
                                loss=BinaryCrossentropy(from_logits=True))
        logging.info("Training Model for Criteria 2 Started")
        # Train the model
        model_criteria2.fit(X_train_criteria2,
                            y_train_criteria2, epochs=50, batch_size=32)
        logging.info("Training Model for Criteria 2 Completed")
        # Save the model
        saved_path = os.path.normpath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "other",
                                                                self.exercise_name, self.evaluation_type, current_time+"_"+self.exercise_name+"_criteria2_"+self.evaluation_type+"_model.h5")))
        model_criteria2.save(saved_path)
        logging.info("Model for Criteria 2 Saved at "+saved_path)

        # Train and save model for criteria 3
        input_shape = (X_train_criteria3.shape[1], X_train_criteria3.shape[2])
        # Create the model
        model_criteria3 = self.create_model(input_shape)
        # Compile the model
        model_criteria3.compile(metrics=['accuracy'],
                                optimizer=Adam(learning_rate=0.0001),
                                loss=BinaryCrossentropy(from_logits=True))
        logging.info("Training Model for Criteria 3 Started")
        # Train the model
        model_criteria3.fit(X_train_criteria3,
                            y_train_criteria3, epochs=50, batch_size=32)
        logging.info("Training Model for Criteria 3 Completed")
        # Save the model
        saved_path = os.path.normpath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "other",
                                                                self.exercise_name, self.evaluation_type, current_time+"_"+self.exercise_name+"_criteria3_"+self.evaluation_type+"_model.h5")))
        model_criteria3.save(saved_path)
        logging.info("Model for Criteria 3 Saved at "+saved_path)



if __name__ == "__main__":
    trainer = ModelTrainer("bicep", "angles")
    trainer.train()
