import torch
import os
from DataProcessing import *
from Config import *
from MyModels import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import models

# ======== Model Architecture ========
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features  # Get the number of input features for the classifier
model.classifier[1] = torch.nn.Linear(in_features, 4)  # Adjust to match your saved model (4 classes)

# ======== Loading Saved Model ========
model.load_state_dict(torch.load(r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Saved Models\MobileNet_V2_1.pth'))
model.to(device)
model.eval()

# Get class_to_idx from the ImageFolder dataset
class_to_idx = train_loader.dataset.dataset.class_to_idx
# Invert the dictionary to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Get a random batch of images
images, labels = next(iter(train_loader))

# Choose a random image from the batch
idx = random.randint(0, len(images) - 1)
image, label = images[idx], labels[idx]

# Make a prediction
image = image.unsqueeze(0).to(device)
with torch.no_grad():
    prediction = model(image)
    predicted_label = prediction.argmax(dim=1).item()

# Convert image tensor to numpy for displaying
image = image.squeeze(0).cpu().numpy()
image = np.transpose(image, (1, 2, 0))  # Rearrange dimensions to height x width x channels
image = (image * 0.5) + 0.5  # Undo normalization
image = np.clip(image, 0, 1)

# Display and save the image
plt.imshow(image)
plt.title(f"Actual: {idx_to_class[label.item()]}, Predicted: {idx_to_class[predicted_label]}")
plt.axis('off')

# Define the save path
save_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Inference Images'
save_filename = f"Predicted by ALexNet Actual_{idx_to_class[label.item()]} Predicted_{idx_to_class[predicted_label]}.png"

# Create the directory if it does not exist
os.makedirs(save_path, exist_ok=True)

# Full path for saving the image
full_save_path = os.path.join(save_path, save_filename)

plt.savefig(full_save_path)
plt.show()
