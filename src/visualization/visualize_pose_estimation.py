import sys
import os

REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

import cv2
import mediapipe as mp
from src.exception import CustomException
from src.logger import logging
import argparse
from src.utils import load_config


class VisualizePoseEstimation:
    """
    This class will visualize the pose estimation.
    input:
        video_source: video source(webcam/video file path)
    methods:
        load_config: This function will load the yaml configuration file.
        visualize_pose_estimation: This function will visualize the pose estimation.
    """

    def __init__(self, video_source):
        logging.info("Pose Estimation Visualization Started")
        self.config = load_config()
        self.min_detection_confidence = self.config["min_detection_confidence"]
        self.min_tracking_confidence = self.config["min_tracking_confidence"]
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_model = mp.solutions.pose.Pose(
            min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        self.video_source = video_source

    def visualize(self):
        """
        This function will visualize the pose estimation.
        """
        try:
            if self.video_source == "webcam":
                logging.info("Webcam Feed Started")
                cap = cv2.VideoCapture(0)
            else:
                logging.info("Video Feed Started")
                cap = cv2.VideoCapture(self.video_source)
            # Setup mediapipe instance
            with self.mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    # Break the loop if there is no frame to read
                    if not ret:
                        logging.error("Error: Unable to access video feed")
                        break

                    # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    # Make detection
                    results = pose.process(image)

                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Continue if no pose landmarks are detected
                    if results.pose_landmarks == None:
                        continue

                    # Render detections
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                   self.mp_drawing.DrawingSpec(
                                                       color=(245, 117, 66), thickness=2, circle_radius=2),
                                                   self.mp_drawing.DrawingSpec(
                                                       color=(245, 66, 230), thickness=2, circle_radius=2)
                                                   )
                    cv2.imshow('Mediapipe Pose Estimation Model Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                logging.info("Pose Estimation Visualization Completed")
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Visualize pose estimation")
        parser.add_argument("source", choices=[
                            "webcam", "video"], help="Input source: 'webcam' for live webcam feed or 'video' for video file path")
        parser.add_argument(
            "--path", type=str, help="Path to the video file (required if source is 'video')")
        args = parser.parse_args()

        # Initialize VisualizePoseEstimation object based on input source
        if args.source == "video":
            if not args.path:
                parser.error(
                    "--path argument is required when source is 'video'")
            else:
                visualizer = VisualizePoseEstimation(args.path)
        else:
            visualizer = VisualizePoseEstimation("webcam")

        # Visualize pose estimation
        visualizer.visualize()
    except Exception as e:
        logging.error("Error: "+str(e))
        raise CustomException(e, sys)
