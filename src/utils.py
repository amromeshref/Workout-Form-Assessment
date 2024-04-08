import sys
import os

REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

import yaml
from src.logger import logging
from src.exception import CustomException


CONFIG_FILE_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yml")))


def load_config():
    """
    This function will load the yaml configuration file.
    input: 
        file_path: yaml file path
    output:
        config: configuration file
    """
    try:
        with open(CONFIG_FILE_PATH) as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logging.error("Error: "+str(e))
        raise CustomException(e, sys)


    