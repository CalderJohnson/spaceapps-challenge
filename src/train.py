"""Training loop for the model."""
import torch
from torch.utils.data import DataLoader

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
train_loader = DataLoader(preprocessor.get_training_data(), batch_size=32)

for epoch in range(config.N_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
    
    # Print loss every epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print("Training complete.")
