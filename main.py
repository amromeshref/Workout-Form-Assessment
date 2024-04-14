import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_DIR_PATH)

from pydantic import BaseModel
from fastapi import FastAPI
from src.utils import load_config
from src.models.predict_model import ModelPredictor
from src.data_preprocessing.cycles_divider import CyclesDivider
from datetime import datetime
import uvicorn



# Load configuration file
CONFIG_FILE = load_config()

# Get the evaluation type from the configuration
EVALUATION_TYPE = CONFIG_FILE['best_evaluation_type']

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class FeedbackModelInput(BaseModel):
    video_path: str
    exercise_name: str

# Define endpoint to get feedback
@app.post("/get-feedback")
def get_feedback(input_data: FeedbackModelInput):
    # Extract input data
    video_path = input_data.video_path
    exercise_name = input_data.exercise_name

    # Initialize CyclesDivider and ModelPredictor
    divider = CyclesDivider(exercise_name, EVALUATION_TYPE)
    predictor = ModelPredictor(exercise_name, EVALUATION_TYPE)

    # Get cycles from the video
    cycles = divider.get_cycles(video_path)

    # Create output directory
    current_time = str(datetime.now().strftime('%Y-%m-%d-%I-%M-%S'))
    output_dir = os.path.normpath(os.path.join(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "feedback", current_time)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save cycles as videos
    divider.save_cycles_as_videos(cycles, current_time, output_dir)

    # Get sequential data from cycles and predict feedback for each sequence
    sequential_data = divider.get_sequential_data(cycles)
    cnt = 1
    results = {}
    feedback = {}
    for sequence in sequential_data:
        prediction1 = predictor.predict_criteria1(sequence)
        prediction2 = predictor.predict_criteria2(sequence)
        prediction3 = predictor.predict_criteria3(sequence)
        feedback["Cycle "+str(cnt)] = {
            "Criteria 1": prediction1,
            "Criteria 2": prediction2,
            "Criteria 3": prediction3
        }
        cnt += 1
    
    results["feedback"] = feedback
    results["saved_videos_dir"] = output_dir

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)