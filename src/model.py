"""Quake detection model definition for the Space Apps 2024 project."""
import torch
import torch.nn as nn

class QuakeDetector(torch.nn.Module):
    """Time series transformer model for quake detection."""
    def __init__(self, input_size, d_model, num_heads, num_layers, output_size, max_seq_len, dropout=0.1):
        super(QuakeDetector, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, d_model)
        self.bn_embedding = nn.BatchNorm1d(d_model)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)  # max sequence length
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer (final prediction)
        self.fc = nn.Linear(d_model, output_size)
        self.relu = nn.ReLU()
    
    def create_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, inference=False):
        x = self.embedding(x)
        batch_size, seq_len, d_model = x.size()
        x = x.view(-1, d_model)  # Shape: [batch_size * seq_len, d_model]
        x = self.bn_embedding(x)  # Normalizing across the feature dimension
        x = x.view(batch_size, seq_len, d_model)  # Shape: [batch_size, seq_len, d_model]
        x += self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])  # Prediction from the last time step
        out = self.relu(out)  # Ensure outputs are non-negative
        return out
