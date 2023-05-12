import torch
import torch.nn as nn
from .mnist_model import MNISTModel

def print_weights_biases(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            print(f"Layer {name}:")
            print("Weights:")
            print(layer.weight.data)
            print("Biases:")
            print(layer.bias.data)

# To use the function, just call it with your model:

model = MNISTModel()

print_weights_biases(model)
