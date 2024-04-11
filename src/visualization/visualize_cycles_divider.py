import argparse
from datetime import datetime
import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.append(REPO_DIR_PATH)

from src.data_preprocessing.cycles_divider import CyclesDivider
from src.logger import logging
from src.exception import CustomException
import cv2


class VisualizeCyclesDivider(CyclesDivider):
    """
    This class will visualize the cycles divider.
    input:
        video_source: video source(webcam/video file path)
    methods:
        visualize_cycles_divider: This function will visualize the cycles divider.
    """

    def __init__(self, exercise_name, video_source):
        super().__init__(exercise_name, "angles")
        self.video_source = video_source

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

            if not cap.isOpened():
                logging.error("Error: Cannot open video file")
                raise CustomException("Cannot open video file", sys)
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
                    if self.check_visibility(landmarks, self.divider_landmarks["left"]):
                        first = [landmarks[self.divider_landmarks["left"][0]].x,
                                landmarks[self.divider_landmarks["left"][0]].y]
                        mid = [landmarks[self.divider_landmarks["left"][1]].x,
                            landmarks[self.divider_landmarks["left"][1]].y]
                        end = [landmarks[self.divider_landmarks["left"][2]].x,
                            landmarks[self.divider_landmarks["left"][2]].y]
                    else:
                        first = [landmarks[self.divider_landmarks["right"][0]].x,
                                landmarks[self.divider_landmarks["right"][0]].y]
                        mid = [landmarks[self.divider_landmarks["right"][1]].x,
                            landmarks[self.divider_landmarks["right"][1]].y]
                        end = [landmarks[self.divider_landmarks["right"][2]].x,
                            landmarks[self.divider_landmarks["right"][2]].y]

                    # Calculate angle and append to the angles list
                    angle = self.calculate_angle(first, mid, end)
                    angles.append(angle)
                except Exception as e:
                    logging.error("Error: "+str(e))
                    raise CustomException(e, sys)
                
                # Display the frame
                cv2.imshow("Cycles Divider", frame)  

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

    def visualize(self):
        """
        This function will visualize the cycles divider.
        input:
            None
        output:
            None
            Cycle videos will be saved in the results/cycles_divider directory
        """
        try:
            # Get the cycles from the video
            cycles = self.get_cycles(self.video_source)
            # Save the cycles as videos
            if self.video_source == 0:
                current_time = str(datetime.now().strftime('%Y-%m-%d-%I-%M-%S'))
                video_name = "webcam_"+current_time
            else:
                video_name = self.get_video_name_without_extension(self.video_source)
            # Define the output directory
            output_dir = os.path.normpath(os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cycles_divider", video_name)))
            # Check if the directory already exists
            if not os.path.exists(output_dir):
                # Create the directory
                os.makedirs(output_dir)
            self.save_cycles_as_videos(cycles, video_name, output_dir)
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Divide the video into cycles")
    parser.add_argument("exercise_name", type=str, help="Exercise name")
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
            visualizer = VisualizeCyclesDivider(args.exercise_name, args.path)
    else:
        visualizer = visualizer = VisualizeCyclesDivider(args.exercise_name, 0)
    
    visualizer.visualize()