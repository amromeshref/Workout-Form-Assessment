import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)


from src.data_preprocessing.cycles_divider import CyclesDivider
from src.logger import logging
from src.exception import CustomException
from src.utils import load_config

class ExternalToInterimTransformer:
    def __init__(self):
        logging.info("Initializing ExternalToInterimTransformer")
        # Load the available_exercises from configuration file
        self.config = load_config()
        self.available_exercises = self.config["available_exercises"]

    def transform(self):
        """
        This function will transform the external data to interim data.
        input:
            None
        output:
            None
        """
        try:
            # Define the path to the external data directory
            external_data_dir = os.path.normpath(os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "external", "self_collected_data"))) + "/"

            # Define the path to the interim data directory
            interim_data_dir = os.path.normpath(os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "interim"))) + "/"

            # Define the list of criterias
            criterias = ["criteria_1", "criteria_2", "criteria_3", "criteria_4"]

            # Iterate over each available exercise
            for exercise in self.available_exercises:
                # Define the directory for the exercise data
                exercise_data_dir = external_data_dir + exercise + "/"

                # Create a CyclesDivider object
                divider = CyclesDivider(exercise, "angles")

                # Iterate over each criteria
                for criteria in criterias:
                    # Define the directory for the criteria data
                    criteria_data_dir = exercise_data_dir + criteria + "/"

                    # Iterate over each label in the criteria directory
                    for label in os.listdir(criteria_data_dir):
                        # Iterate over each video file in the label directory
                        for video_file in os.listdir(criteria_data_dir + label):
                            # Define the full path to the video file
                            video_path = criteria_data_dir + label + "/" + video_file

                            # Log the start of video processing
                            logging.info(
                                "Started processing the video: " + video_path)

                            # Get the cycles from the video
                            cycles = divider.get_cycles(video_path)

                            # Get the video name without extension
                            video_name = divider.get_video_name_without_extension(
                                video_path)

                            # Define the output directory for saving cycles
                            output_dir = interim_data_dir + exercise + "/" + criteria + "/" + label

                            # Save the cycles as videos in the interim directory
                            divider.save_cycles_as_videos(
                                cycles, video_name, output_dir)

                            # Log the completion of video processing
                            logging.info(
                                "Completed processing the video: " + video_path)

            # Log the completion of data transformation
            logging.info(
                "Data transformation from external to interim is completed")
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = ExternalToInterimTransformer()
    transformer.transform()
