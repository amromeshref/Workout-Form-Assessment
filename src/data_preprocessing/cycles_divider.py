import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

import numpy as np
import argparse
from src.logger import logging
from src.exception import CustomException
from src.data_preprocessing.data_transformation import DataTransformer
import cv2
from src.utils import load_config
from scipy.signal import medfilt
from scipy.signal import find_peaks


class CyclesDivider(DataTransformer):
    def __init__(self, exercise_name: str, evaluation_type: str):
        """
        Initialize the CyclesDivider object.
        input:
            exercise_name(str): Exercise name
            evaluation_type(str): Evaluation type
        Methods:
            calculate_angle: Calculate the angle between three points
            get_video_frames_and_angles: Get the frames and angles from the video
            median_filter: Apply median filter to the angles
            get_peaks: Get the peaks from the graph of the angles over frames
            get_troughs: Get the troughs from the graph of the angles over frames
            get_cycles: Get the cycles(as frames) from the video by the logic of increasing and decreasing angles
            save_cycle_frames_as_video: Save a list of frames as a video file
            save_cycles_as_videos: Save the cycles as videos in the output directory
            get_video_name_without_extension: Extract the video name without the file extension from the given video path
        """
        logging.info("Cycle Divider started for exercise: " +
                     exercise_name+", evaluation type: "+evaluation_type)
        # Call the constructor of the parent class
        super().__init__(exercise_name, evaluation_type)

        # Load configuration
        self.config = load_config()

        # Load parameters from the configuration
        self.min_detection_confidence = self.config["min_detection_confidence"]
        self.min_tracking_confidence = self.config["min_tracking_confidence"]

        # Define divider landmarks for the exercise
        self.divider_landmarks = self.config[self.exercise_name +
                                             "_divider_landmarks"]

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points
        input:
            a(list): First point
            b(list): Mid point
            c(list): End point
        output:
            angle(float): Angle between the points in degrees
        """
        try:
            # Convert points to numpy arrays
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            # Calculate angle in radians
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
            # Convert radians to degrees
            angle = np.abs(radians * 180.0 / np.pi)

            # Ensure the angle is within [0, 180) degrees
            if angle > 180.0:
                angle = 360 - angle

            return angle
        except Exception as e:
            logging.error("Error: " + str(e))
            raise CustomException(e, sys)

    def get_video_frames_and_angles(self, video_path: str):
        """
        This function will return the frames and angles from the video.
        input:
            video_path: Video file path
        output:
            frames(list): Video frames
            angles(list): Angles of the video. each angle is between left shoulder, elbow and wrist
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            frames = []  # List to store video frames
            angles = []  # List to store angles

            # Loop through each frame of the video
            while cap.isOpened():
                ret, frame = cap.read()

                # If no frame is retrieved, break the loop
                if not ret:
                    break
                frames.append(frame)  # Append the frame to the frames list

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = self.pose_model.process(image)

                # Extract landmarks and calculate angles
                try:
                    if results.pose_landmarks is None:
                        continue
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates of landmarks
                    first = [landmarks[self.divider_landmarks[0]].x,
                             landmarks[self.divider_landmarks[0]].y]
                    mid = [landmarks[self.divider_landmarks[1]].x,
                           landmarks[self.divider_landmarks[1]].y]
                    end = [landmarks[self.divider_landmarks[2]].x,
                           landmarks[self.divider_landmarks[2]].y]

                    # Calculate angle and append to the angles list
                    angle = self.calculate_angle(first, mid, end)
                    angles.append(angle)
                except Exception as e:
                    logging.error("Error: "+str(e))
                    raise CustomException(e, sys)

                # Exit loop if 'q' key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Release video capture and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

            return frames, angles  # Return the frames and angles lists
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def median_filter(self, angles: list):
        """
        This function will apply median filter to the angles.
        input:
            angles: Angles list
        output:
            denoised_angles(np.array): Denoised angles
        """
        try:
            window_size = 11
            denoised_angles = medfilt(angles, kernel_size=window_size)
            return denoised_angles
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_peaks(self, angles: list):
        """
        This function will return the peaks from the graph of the angles over frames.
        input:
            angles: Angles list
        output:
            peaks(np.array): Peaks in the graph
        """
        try:
            peaks, _ = find_peaks(angles, prominence=0.1)
            return peaks
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_troughs(self, angles: list):
        """
        This function will return the troughs from the graph of the angles over frames.
        input:
            angles: Angles list
        output:
            peaks(np.array): Peaks in the graph
        """
        try:
            troughs, _ = find_peaks(-angles, prominence=0.1)
            return troughs
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_cycles(self, video_path: str):
        """
        This function will return the cycles(as frames) from the video by the logic of increasing and decreasing angles.
        input:
            peaks: Peaks list
        output:
            cycles(list): Cycles of the video
        """
        try:
            # Get frames and angles from the video
            frames, angles = self.get_video_frames_and_angles(video_path)
            logging.info("Frames and angles extracted successfully from the video")

            # Apply median filter to noisy angles
            denoised_angles = self.median_filter(angles)

            # Get peaks based on exercise type
            # If exercise is bicep, get peaks
            # If exercise is lateral_raise, get troughs
            peaks = None
            if self.exercise_name == "bicep":
                peaks = self.get_peaks(denoised_angles)
            elif self.exercise_name == "lateral_raise":
                peaks = self.get_troughs(denoised_angles)

            # Divide the video into cycles based on the peaks
            cycles = []
            for peak_index in range(len(peaks)-1):
                # Get the frames between two peaks
                cnt = peaks[peak_index]
                cycle = []
                while True:
                    # Append frames to the cycle until the next peak
                    cycle.append(frames[cnt])
                    # Break the loop if the next peak is reached
                    if cnt == peaks[peak_index+1]:
                        break
                    cnt += 1
                # Append the cycle to the cycles list
                cycles.append(cycle)
            return cycles
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def save_cycle_frames_as_video(self, cycle_frames, output_path, fps=24):
        """
        Save a list of frames as a video file.
        input:
            cycle_frames(list): List of frames
            output_path(str): Output video file path
            fps(int): Frames per second
        output:
            None
        """
        try:
            # Get the shape of the first frame to determine video dimensions
            height, width, _ = cycle_frames[0].shape

            # Define the codec and create VideoWriter object
            # Choose the codec (here, MP4V)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            try:
                # Write each frame to the video file
                for frame in cycle_frames:
                    out.write(frame)
            finally:
                # Release the VideoWriter object
                out.release()
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def save_cycles_as_videos(self, cycles: list, video_name: str, output_dir):
        """
        This function will save the cycles as videos in the output directory.
        input:
            cycles(list): List of cycles
            video_name(str): Video name
        output:
            None
        """
        try:
            for i, cycle in enumerate(cycles):
                output_path = os.path.join(
                    output_dir, video_name+"_cycle"+str(i+1)+".mp4")
                self.save_cycle_frames_as_video(cycle, output_path)
            logging.info("Cycles saved as videos successfully for the video: " +
                         video_name + " at: " + output_dir)
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def get_video_name_without_extension(self, video_path):
        """
        Extract the video name without the file extension from the given video path.
        input:
            video_path(str): Video file path
        output:
            video_name(str): Video name without extension
        """
        try:
            # Get the base name of the video path (i.e., the file name with extension)
            video_name_with_extension = os.path.basename(video_path)
            # Split the base name into the name and the extension
            video_name, _ = os.path.splitext(video_name_with_extension)
            return video_name
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

