import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

import argparse
from src.logger import logging
from src.exception import CustomException
from src.utils import load_config
import cv2
import mediapipe as mp
import numpy as np


class DataTransformer:
    """
    This class will transform the video data into the required format for the model training.
    input:
        exercise_name: exercise name
        evaluation_type: evaluation type(poses/lengths/angles)
    methods:
        load_config: This function will load the yaml configuration file.
        get_video_poses: This function will return the required poses of the exercise in each frame in the video.
        get_video_angles: This function will return the required angles of the exercise in each frame in the video.
        calculate_angle: This function will calculate the angle between the three points.
        get_training_data_per_criteria: This function will return the training data for each criteria.
        prepare_training_data: This function will prepare the training data for the model training.
    """

    def __init__(self, exercise_name: str, evaluation_type: str):
        logging.info("Data Transformation Started for exercise: " +
                     exercise_name+", evaluation type: "+evaluation_type)
        # Initialize the class with exercise name and evaluation type
        self.exercise_name = exercise_name
        self.evaluation_type = evaluation_type
        # Load configuration from YAML file
        self.config = load_config()
        # Load available evaluation types and exercises from configuration
        self.available_evaluation_types = self.config["available_evaluation_types"]
        self.available_exercises = self.config["available_exercises"]
        # Check if the exercise name is available in the configuration
        if self.exercise_name not in self.available_exercises:
            # Raise an error if the exercise name is not available
            logging.error("Exercise Name Not Supported or Not Found")
            raise CustomException(
                "Exercise Name Not Supported or Not Found", sys)
        # Check if the evaluation type is available in the configuration
        if self.evaluation_type not in self.available_evaluation_types:
            # Raise an error if the evaluation type is not available
            logging.error("Evaluation Type Not Supported or Not Found")
            raise CustomException(
                "Evaluation Type Not Supported or Not Found", sys)
        # Load poses and angles landmarks from configuration
        self.poses_landmarks = self.config[exercise_name+"_poses_landmarks"]
        self.angles_landmarks = self.config[exercise_name+"_angles_landmarks"]
        # Load parameters from configuration
        self.max_frames_number_per_cycle = self.config["max_frames_number_per_cycle"]
        self.min_detection_confidence = self.config["min_detection_confidence"]
        self.min_tracking_confidence = self.config["min_tracking_confidence"]
        # Initialize MediaPipe Pose model with parameters
        self.pose_model = mp.solutions.pose.Pose(
            min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)

    def calculate_angle(self, start: list, mid: list, end: list):
        """
        This function will calculate the angle between the three points.
        input:
            start: start point[x,y]
            mid: mid point[x,y]
            end: end point[x,y]
        output:
            angle: angle between the three points in degrees
        """
        try:
            # Convert points to numpy arrays
            start = np.array(start)
            mid = np.array(mid)
            end = np.array(end)
            # Calculate radians
            radians = np.arctan2(end[1]-mid[1], end[0]-mid[0]) - \
                np.arctan2(start[1]-mid[1], start[0]-mid[0])
            # Convert radians to degrees
            angle = np.abs(radians*180.0/np.pi)
            # Normalize angle
            if angle > 180.0:
                angle = 360-angle

            return angle
        except Exception as e:
            # Log error and raise exception
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_frame_angles(self, frame):
        """
        This function will return the required angles of the exercise in the frame.
        input: 
            frame: frame(bgr format)
        output: 
            frame_angles: frame angles(list of angles in each frame)
        """
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process frame with pose estimation model
            results = self.pose_model.process(frame_rgb)
            # Return None if no pose landmarks are detected
            if results.pose_landmarks == None:
                return None
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            # Initialize list to store angles for current frame
            frame_angles = []
            # Calculate angles for each angle landmark
            for angle_landmarks in self.angles_landmarks:
                start = [landmarks[angle_landmarks[0]].x,
                         landmarks[angle_landmarks[0]].y]
                mid = [landmarks[angle_landmarks[1]].x,
                       landmarks[angle_landmarks[1]].y]
                end = [landmarks[angle_landmarks[2]].x,
                       landmarks[angle_landmarks[2]].y]
                frame_angles.append(self.calculate_angle(start, mid, end))
            return frame_angles
        except Exception as e:
            # Log error and raise exception
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_video_angles(self, video_path: str):
        """
        This function will return the required angles of the exercise in each frame in the video.
        input: 
            video_path: video file path
        output: 
            video_angles: video angles(list of angles in each frame)
        """
        try:
            # Open video capture object
            video_cap = cv2.VideoCapture(video_path)
            # Initialize list to store angles for each frame
            video_angles = []
            # Loop through video frames
            while True:
                # Read frame
                ret, frame = video_cap.read()
                # Break loop if no frame is retrieved
                if not ret:
                    break
                # Break loop if maximum frames per cycle is reached
                if (len(video_angles) == self.max_frames_number_per_cycle):
                    break
                # Get angles for current frame
                frame_angles = self.get_frame_angles(frame)
                # Continue to next iteration if no pose landmarks are detected
                if frame_angles == None:
                    continue
                # Append angles for current frame to video_angles list
                video_angles.append(frame_angles)
            # If the video is less than the required frames number, fill the rest with zeros
            while (len(video_angles) < self.max_frames_number_per_cycle):
                video_angles.append([0]*len(self.angles_landmarks))
            return [video_angles]
        except Exception as e:
            # Log error and raise exception
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_frame_poses(self, frame):
        """
        This function will return the required poses of the exercise in the frame.
        input: 
            frame: frame(bgr format)
        output: 
            frame_poses: frame poses(list of landmarks x and y coordinates)
        """
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process frame with pose estimation model
            results = self.pose_model.process(frame_rgb)
            # Return None if no pose landmarks are detected
            if results.pose_landmarks == None:
                return None
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            # Initialize list to store poses for current frame
            frame_poses = []
            # Append x and y coordinates of each pose landmark to frame_poses list
            for landmark in self.poses_landmarks:
                frame_poses.append(landmarks[landmark].x)
                frame_poses.append(landmarks[landmark].y)
            return frame_poses
        except Exception as e:
            # Log error and raise exception
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_video_poses(self, video_path: str):
        """
        This function will return the required poses of the exercise in each frame in the video.
        input: 
            video_path: video file path
        output: 
            video_poses: video poses(list of landmarks x and y coordinates in each frame)
        """
        try:
            # Open video capture object
            video_cap = cv2.VideoCapture(video_path)
            # Initialize list to store poses for each frame
            video_poses = []
            # Loop through video frames
            while True:
                # Read frame
                ret, frame = video_cap.read()
                # Break loop if no frame is retrieved
                if not ret:
                    break
                # Break loop if maximum frames per cycle is reached
                if (len(video_poses) == self.max_frames_number_per_cycle):
                    break
                # Get poses for current frame
                frame_poses = self.get_frame_poses(frame)
                # Continue to next iteration if no pose landmarks are detected
                if frame_poses == None:
                    continue
                # Append poses for current frame to video_poses list
                video_poses.append(frame_poses)
            # If the video is less than the required frames number, fill the rest with zeros
            while (len(video_poses) < self.max_frames_number_per_cycle):
                video_poses.append([0]*len(self.poses_landmarks)*2)
            return [video_poses]
        except Exception as e:
            # Log error and raise exception
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_training_data_per_criteria(self, criteria: str):
        """
        This function will return the training data for each criteria.
        input:
            criteria: criteria
        output:
            training_data: training data 
        """
        try:
            # Initialize an empty list to store training data
            training_data = []
            # Define the directory path where the videos are stored based on the exercise name and criteria
            dir_path = os.path.normpath(os.path.join(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/interim/")))
            dir_path += "/"+self.exercise_name+"/"+criteria

            # Initialize the data size
            data_size = 0
            for label in os.listdir(dir_path):
                data_size += len(os.listdir(dir_path+"/"+label))    
            
            # Initialize the counter
            cnt = 0
            # Initialize the conditions for logging
            cond_25 = True
            cond_50 = True
            cond_75 = True
            # Loop through each label in the directory
            for label in os.listdir(dir_path):
                # Loop through each video file in the label directory
                for video_file in os.listdir(dir_path+"/"+label):
                    # Get the full path of the video file
                    video_path = dir_path+"/"+label+"/"+video_file
                    # Check the evaluation type to determine how to process the video data
                    if self.evaluation_type == "poses":
                        # Get the poses from the video
                        video_poses = self.get_video_poses(video_path)
                        # Append the video poses and label to the training data list
                        training_data.append((video_poses, int(label)))
                    elif self.evaluation_type == "angles":
                        # Get the angles from the video
                        video_angles = self.get_video_angles(video_path)
                        # Append the video angles and label to the training data list
                        training_data.append((video_angles, int(label)))
                    else:
                        # Raise an error if the evaluation type is not supported
                        logging.error(
                            "Evaluation Type Not Supported or Not Found")
                        raise CustomException(
                            "Evaluation Type Not Supported or Not Found", sys)
                    if cnt/data_size >= 0.25 and cond_25:
                        logging.info("Data Transformation Progress: 25%")
                        cond_25 = False
                    if cnt/data_size >= 0.5 and cond_50:
                        logging.info("Data Transformation Progress: 50%")
                        cond_50 = False
                    if cnt/data_size >= 0.75 and cond_75:
                        logging.info("Data Transformation Progress: 75%")
                        cond_75 = False
                    cnt += 1
            # Convert the training data list to a numpy array
            training_data = np.array(training_data, dtype=object)
            # Return the training data
            return training_data
        except Exception as e:
            # Log and raise an exception if an error occurs
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def prepare_training_data(self):
        """
        This function will prepare the training data for the model training, and save it in the processed folder.
        input:
            None
        output:
            None
        """
        try:
            # Loop through each criteria in the interim data directory
            dir_path = os.path.normpath(os.path.join(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/interim")))
            dir_path += "/"
            for criteria in os.listdir(dir_path+self.exercise_name):
                # Log the start of data transformation for the current criteria
                logging.info("Data Transformation Started for "+criteria + ", exercise: " +
                             self.exercise_name + ", evaluation type: "+self.evaluation_type)
                # Get the training data for the current criteria
                training_data = self.get_training_data_per_criteria(criteria)
                # Shuffle the training data
                np.random.shuffle(training_data)
                # Save the training data in the processed folder
                saved_path = os.path.normpath(os.path.join(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/processed")))
                saved_path += "/" + self.exercise_name+"/" + self.evaluation_type + \
                    "/" + "training_data_"+self.exercise_name+"_"+self.evaluation_type+"_"+criteria
                np.save(saved_path, training_data)
                # Log the completion of data transformation for the current criteria
                logging.info("Data Transformation Completed for "+criteria + ", exercise: " +
                             self.exercise_name + ", evaluation type: "+self.evaluation_type)
                logging.info("Training Data Saved for "+criteria+", exercise: "+self.exercise_name +
                             ", evaluation type: "+self.evaluation_type+" at "+saved_path+".npy")

            # Log the completion of data transformation
            logging.info("Data Transformation Completed for exercise: " +
                         self.exercise_name+", evaluation type: "+self.evaluation_type)
        except Exception as e:
            # Log and raise an exception if an error occurs
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def get_sequential_data(self, cycles: list):
        """
        This function will convert the cycles in the video into sequential data.
        input:
            cycles: cycles
        output:
            seq_data(list): sequential data
        """
        try:
            # Check the evaluation type
            if self.evaluation_type == "poses":
                # Initialize an empty list to store sequential data
                seq_data = []
                # Iterate over each cycle in the cycles list
                for cycle in cycles:
                    # Initialize an empty list to store poses for each frame in the cycle
                    seq = []
                    # Iterate over each frame in the cycle
                    for frame in cycle:
                        # Get poses for the current frame
                        frame_poses = self.get_frame_poses(frame)
                        # If no poses are detected, continue to the next frame
                        if frame_poses is None:
                            continue
                        # Append the poses for the current frame to the seq list
                        seq.append(frame_poses)
                    # If the video is less than the required frames number, fill the rest with zeros
                    while (len(seq) < self.max_frames_number_per_cycle):
                        seq.append([0]*len(self.poses_landmarks)*2)
                    # Convert the seq list to a numpy array and append it to the seq_data list
                    seq_data.append(np.array(seq))
                # Return the sequential data
                return seq_data
            elif self.evaluation_type == "angles":
                # Initialize an empty list to store sequential data
                seq_data = []
                # Iterate over each cycle in the cycles list
                for cycle in cycles:
                    # Initialize an empty list to store angles for each frame in the cycle
                    seq = []
                    # Iterate over each frame in the cycle
                    for frame in cycle:
                        # Break loop if maximum frames per cycle is reached
                        if (len(seq) == self.max_frames_number_per_cycle):
                            break
                        # Get angles for the current frame
                        frame_angles = self.get_frame_angles(frame)
                        # If no angles are detected, continue to the next frame
                        if frame_angles is None:
                            continue
                        # Append the angles for the current frame to the seq list
                        seq.append(frame_angles)
                    # If the video is less than the required frames number, fill the rest with zeros
                    while (len(seq) < self.max_frames_number_per_cycle):
                        seq.append([0]*len(self.angles_landmarks))
                    # Convert the seq list to a numpy array and append it to the seq_data list
                    seq_data.append(np.array(seq))
                # Return the sequential data
                return seq_data
        except Exception as e:
            # Log and raise any exceptions that occur
            logging.error("Error: " + str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Transform video data for model training")
    parser.add_argument("exercise_name", type=str, help="Exercise name")
    parser.add_argument("evaluation_type", type=str,
                        help="Evaluation type (poses/lengths/angles)")
    args = parser.parse_args()

    # Create DataTransformer object with command-line arguments
    transformer = DataTransformer(args.exercise_name, args.evaluation_type)
    transformer.prepare_training_data()
