import torch.nn as nn
import torch.optim
from TrainSteps import *
from DataProcessing import *
from torchvision.models import vgg16, VGG16_Weights

# ================= Paths for Saving Model & Training Results =================
model_path = r"E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Saved Models\VGG16_1.pth"
results_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Training Results\VGG16_1_results.pkl'


VGG16 = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

for params in VGG16.parameters():
    params.requires_grad = False

# for params in VGG16.features[30].parameters():
#     params.requires_grad = True

# Unfreeze the last classifier layer
for params in VGG16.classifier[6].parameters():
    params.requires_grad = True

in_features = VGG16.classifier[6].in_features
VGG16.classifier[6] = nn.Linear(in_features, len(class_names))

torch.manual_seed(42)
torch.cuda.manual_seed(42)
VGG16 = VGG16.to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = VGG16.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Training the model
VGG16_results = train (model = VGG16, train_dataloader = train_loader,
                          val_dataloader = val_loader, optimizer = optimizer,
                          loss_fn = loss_fn, epochs = num_epochs,
                          model_path = model_path)

end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")

# ================= Saving the Training Results =================
import pickle

with open(results_path, 'wb') as file:
    pickle.dump(VGG16_results, file)

print(f"Results saved to {results_path}")

