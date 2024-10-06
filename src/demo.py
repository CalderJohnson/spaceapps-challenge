"""Demonstration of the model"""
import pandas as pd

import config
from interface import get_model_prediction
from plot import create_plot
from preprocessor import preprocess_data

# Plot the quakes in the training data 
CATALOG = pd.read_csv(config.LUNAR_TRAINING_CATALOG)
for _, row in CATALOG.iterrows():
    try:
        file = f"{config.LUNAR_TRAINING_DATA}/{row['filename']}.csv"
        df = pd.read_csv(f"{config.LUNAR_TRAINING_DATA}/{row['filename']}.csv")
    except FileNotFoundError:
        continue
    
    seismic_data, timestamp = preprocess_data(df, row["time_rel(sec)"])
    prediction = get_model_prediction(file)
    print(prediction)
    create_plot(file, prediction)
