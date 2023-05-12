import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from .mnist_model import MNISTModel
from .. import config as conf

processed_data_dir = conf.processed_data_dir

train_dataset = torch.load(os.path.join(processed_data_dir, "train_normalized.pt"))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = torch.load(os.path.join(processed_data_dir, "val_normalized.pt"))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


# Model, loss function, and optimizer
model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.8)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset gradients for each batch
        output = model(data)  # Forward pass: compute predictions
        loss = criterion(output, target)  # Compute the loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Epoch {epoch}, Validation loss: {val_loss}, Validation accuracy: {accuracy:.2f}%")

def next_model_path(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
    next_number = max(model_numbers, default=0) + 1
    return os.path.join(models_dir, f'model_{next_number}.pth')

models_dir = '/Users/ayush/Desktop/Handwritten/Handwritten_digit_recognition/models'
model_path = next_model_path(models_dir)
torch.save(model.state_dict(), model_path)
