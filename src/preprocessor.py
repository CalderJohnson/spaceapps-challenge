"""Data loading and preprocessing functionality."""
import torch
import config
import pandas as pd

PRELOADED_CATALOG = pd.read_csv(config.LUNAR_TRAINING_CATALOG)

def load_next():
    """Generator to yield each CSV files data from the catalog."""
    for _, row in PRELOADED_CATALOG.iterrows():
        df = pd.read_csv(f"{config.LUNAR_TRAINING_DATA}/{row['file']}")

        time_rel = torch.tensor(df['time_rel(sec)'].values, dtype=torch.float32)
        velocity = torch.tensor(df['velocity(m/s)'].values, dtype=torch.float32)
        
        # Stack them into a 2D tensor (shape: [n_samples, 2])
        seismic_data = torch.stack((time_rel, velocity), dim=1)
        
        # Extract the associated timestamp from the catalog
        timestamp = row['time_rel(sec)']
        
        # Yield the tuple (tensor, timestamp)
        yield seismic_data, timestamp

def get_training_data():
    """Load the training data from the catalog."""
    inputs, targets = [], []
    for seismic_data, timestamp in load_next():
        inputs.append(seismic_data), targets.append(timestamp)
    return data
