"""Data loading and preprocessing functionality."""
import torch
import config
import pandas as pd
import numpy as np

PRELOADED_CATALOG = pd.read_csv(config.LUNAR_TRAINING_CATALOG)

def load_next():
    """Generator to yield each CSV files data from the catalog."""
    for _, row in PRELOADED_CATALOG.iterrows():
        try:
            df = pd.read_csv(f"{config.LUNAR_TRAINING_DATA}/{row['filename']}.csv")
        except FileNotFoundError:
            continue

        # Filter out unlikely regions
        df_filtered = df[(df['velocity(m/s)'] < config.VELOCITY_THRESHOLD[0]) | (df['velocity(m/s)'] > config.VELOCITY_THRESHOLD[1])]

        # Downsample to the model's context window
        if len(df_filtered) > config.MAX_SEQ_LEN:
            indices = np.linspace(0, len(df_filtered) - 1, config.MAX_SEQ_LEN, dtype=int)
            df_filtered = df_filtered.iloc[indices]
        
        # Normalize relative time
        df_filtered['time_rel(sec)'] = (df_filtered['time_rel(sec)'] - df_filtered['time_rel(sec)'].min()) / (df_filtered['time_rel(sec)'].max() - df_filtered['time_rel(sec)'].min())

        # Min max normalize seismic velocity
        min_velocity = df_filtered['velocity(m/s)'].min()
        max_velocity = df_filtered['velocity(m/s)'].max()
        df_filtered['velocity(m/s)'] = 2 * (df['velocity(m/s)'] - min_velocity) / (max_velocity - min_velocity) - 1

        # Extract the time_rel and velocity columns
        time_rel = torch.tensor(df_filtered['time_rel(sec)'].values, dtype=torch.float32)
        velocity = torch.tensor(df_filtered['velocity(m/s)'].values, dtype=torch.float32)
        
        # Stack them into a 2D tensor (shape: [n_samples, 2])
        seismic_data = torch.stack((time_rel, velocity), dim=1)
        
        # Extract the associated timestamp from the catalog
        timestamp = row['time_rel(sec)'] / df['time_rel(sec)'].max()
        
        # Yield the tuple (tensor, timestamp)
        yield seismic_data, timestamp

def get_training_data():
    """Load the training data from the catalog."""
    inputs, targets = [], []
    for seismic_data, timestamp in load_next():
        inputs.append(seismic_data), targets.append(timestamp)
    return torch.stack(inputs), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
