"""Training loop for the model."""
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
import preprocessor
from model import QuakeDetector

# Instantiate the model and optimizer
model = QuakeDetector(
    config.INPUT_SIZE,
    config.D_MODEL,
    config.NUM_HEADS,
    config.NUM_LAYERS,
    config.OUTPUT_SIZE,
    config.MAX_SEQ_LEN
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the training data
inputs, targets = DataLoader(preprocessor.get_training_data(), batch_size=32)
