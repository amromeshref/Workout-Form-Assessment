from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.models import load_model
import sys

class ModelPredictor:
    def __init__(self, model_path: str):
        logging.info("Prediction Started")
        self.model_path = model_path
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e,sys)

    def predict(self, input_sequential_data: list):
        predictions = self.model(input_sequential_data)
        logging.info("Prediction Completed")
        return predictions