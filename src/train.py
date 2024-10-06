"""Training loop for the model."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

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
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce learning rate by a factor of 2 every 5 epochs

# Load the training data
inputs, targets = preprocessor.get_training_data()
dataset = TensorDataset(inputs, targets)

# Split the data into training and validation sets
train_size = int(config.SPLIT * len(inputs))
val_size = len(inputs) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print("Training started.")

for epoch in range(config.N_EPOCHS):
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
    
    # Update the learning rate
    scheduler.step()
    
    # Print loss every epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{config.N_EPOCHS}], Loss: {epoch_loss:.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

print("Training complete.")

# Save the model
torch.save(model.state_dict(), "model.pt")
