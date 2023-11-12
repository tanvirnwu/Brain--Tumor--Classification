import torch
import torch.nn as nn

lr = 0.001
batch_size = 32
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_size = 0.2
hidden_units = 32
output_shape = 4
input_shape = 3
