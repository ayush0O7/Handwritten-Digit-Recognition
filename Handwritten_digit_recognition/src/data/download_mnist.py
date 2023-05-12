import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .. import config as conf

processed_data_dir = conf.processed_data_dir
raw_data_dir       = conf.raw_data_dir

# Create the directories if they don't exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Define the normalization transform
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST specific mean and std
])

# Download and normalize the MNIST dataset
train_dataset = datasets.MNIST(raw_data_dir, train=True, download=True, transform=normalize)
test_dataset = datasets.MNIST(raw_data_dir, train=False, download=True, transform=normalize)

# Split the original training data into a new training set and a validation set
train_size = int(0.8 * len(train_dataset))  # 80% of the data for training
val_size = len(train_dataset) - train_size  # 20% of the data for validation
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Save the datasets
torch.save(train_dataset, os.path.join(processed_data_dir, "train_normalized.pt"))
torch.save(val_dataset, os.path.join(processed_data_dir, "val_normalized.pt"))
torch.save(test_dataset, os.path.join(processed_data_dir, "test_normalized.pt"))
