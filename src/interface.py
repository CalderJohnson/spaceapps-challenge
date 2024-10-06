"""Interface with the model for inference"""
import pandas as pd

import config
from preprocessor import preprocess_data
from model import QuakeDetector

model = QuakeDetector(
    config.INPUT_SIZE,
    config.D_MODEL,
    config.NUM_HEADS,
    config.NUM_LAYERS,
    config.OUTPUT_SIZE,
    config.MAX_SEQ_LEN
)

def get_model_prediction(file):
    """Get the model's prediction for a given seismic data file."""
    df = pd.read_csv(file)
    seismic_data, _ = preprocess_data(df)
    seismic_data = seismic_data.unsqueeze(0)  # Add batch dimension
    prediction = model(seismic_data, inference=True)
    return prediction.item() * df['time_rel(sec)'].max()
