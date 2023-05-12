import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)  # Input layer: 28x28 input size (flattened image), 128 nodes in the first hidden layer
        self.layer2 = nn.Linear(128, 64)       # Hidden layer: 128 input nodes (from the previous layer), 64 nodes in this layer
        self.layer3 = nn.Linear(64, 10)        # Output layer: 64 input nodes (from the previous layer), 10 output nodes (one for each digit)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input tensor
        x = F.relu(self.layer1(x)) # First hidden layer with ReLU activation
        x = F.relu(self.layer2(x)) # Second hidden layer with ReLU activation
        x = self.layer3(x)         # Output layer with raw scores (logits)
        return x

model = MNISTModel()
