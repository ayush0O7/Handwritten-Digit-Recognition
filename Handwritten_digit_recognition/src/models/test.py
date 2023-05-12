import os
import torch
from .mnist_model import MNISTModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from .. import config as conf

processed_data_dir = conf.processed_data_dir

# Load the saved model
model_path = '/Users/ayush/Desktop/Handwritten/Handwritten_digit_recognition/models/model_2.pth'
model = MNISTModel()
model.load_state_dict(torch.load(model_path))

# Load the test dataset
test_dataset = torch.load(os.path.join(processed_data_dir, "test_normalized.pt"))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Testing"):
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_accuracy = 100 * correct / total
print(f'Test accuracy: {test_accuracy:.2f}%')
