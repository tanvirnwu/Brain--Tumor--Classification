from Config import *
from DataProcessing import *
from TrainSteps import *
from torchvision.models import alexnet, AlexNet_Weights

# ================= Paths for Saving Model & Training Results =================
model_path = "E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Saved Models\AlexNet_1.pth"
results_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Training Results\AlexNet_1_results.pkl'

# ================= Loading AlexNet Model's Pretrained Weights  =================
AlexNet_1 = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

# ================ Freezing Layers of AlexNet Model =================
# Freeze all layers in the model
for param in AlexNet_1.parameters():
    param.requires_grad = False

# Unfreeze the last classifier layer
for param in AlexNet_1.classifier[6].parameters():
    param.requires_grad = True

# ================= Modifying AlexNet Model's Classifier Layer =================
in_features = AlexNet_1.classifier[6].in_features
AlexNet_1.classifier[6] = nn.Linear(in_features, len(class_names))

# summary(AlexNet_1, (3, 224, 224))

# ================= Training MobileNetV2 =================
torch.manual_seed(42)
torch.cuda.manual_seed(42)
AlexNet_1 = AlexNet_1.to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = AlexNet_1.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Training the model
AlexNet_1_results = train(model=AlexNet_1, train_dataloader = train_loader,
                        val_dataloader = val_loader, optimizer = optimizer,
                        loss_fn = loss_fn, epochs = num_epochs,
                          model_path = model_path)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")



# ================= Saving the Training Results =================
import pickle

with open(results_path, 'wb') as file:
    pickle.dump(AlexNet_1_results, file)

print(f"Results saved to {results_path}")


# ================= Loading the Training Results =================
# with open(results_path, 'rb') as file:
#     loaded_results = pickle.load(file)
# print(loaded_results)
