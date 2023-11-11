import torch
import torch.nn as nn
import torchvision
from Config import *
from DataProcessing import *
from MyModels import *
from TrainSteps import *
from PlotLossCurve import *

torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Training'
test_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Testing'
train_loader, val_loader, test_loader = data_preparation(train_path, test_path, batch_size)

# Since we used random_split, access the original dataset using .dataset
classes = train_loader.dataset.dataset.classes

# Recreate an instance of model 1
mymodel_1 = MyModel_1(input_shape=3, hidden_units=10, output_shape=len(classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = mymodel_1.parameters(), lr=0.001)
model_path = "E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Saved Models\mymodel_1.pth"
# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_0
mymodel_1_results = train(model=mymodel_1, train_dataloader = train_loader,
                        val_dataloader = val_loader, optimizer = optimizer,
                        loss_fn = loss_fn, epochs = num_epochs,
                          model_path = model_path)


end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

plot_loss_curves(mymodel_1_results)
