import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

from src.data_preprocessing.data_transformation import DataTransformer


if __name__ == "__main__":
    transformer = DataTransformer("bicep", "poses")
    transformer.prepare_training_data()
    transformer = DataTransformer("bicep", "angles")
    transformer.prepare_training_data()
    transformer = DataTransformer("lateral_raise", "poses")
    transformer.prepare_training_data()
    transformer = DataTransformer("lateral_raise", "angles")
    transformer.prepare_training_data()