from TrainSteps import *
from DataProcessing import *
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ================= Loading MobileNetV2 Model's Pretrained Weights  =================
mobilenet_v2 = mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)

# ================= Modifying MobileNetV2 Model's Classifier Layer =================
in_features = mobilenet_v2.classifier[1].in_features
mobilenet_v2.classifier[1] = nn.Linear(in_features, len(class_names))

# summary(mobilenet_v2, (3, 224, 224))

# ================= Training MobileNetV2 =================
torch.manual_seed(42)
torch.cuda.manual_seed(42)
mobilenet_v2 = mobilenet_v2.to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = mobilenet_v2.parameters(), lr=0.001)
model_path = "E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Saved Models\MobileNet_V2_1.pth"

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Training the model
MobileNet_V2_Results = train(model=mobilenet_v2, train_dataloader = train_loader,
                        val_dataloader = val_loader, optimizer = optimizer,
                        loss_fn = loss_fn, epochs = num_epochs,
                          model_path = model_path)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")



# ================= Saving the Training Results =================
import pickle
results_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Training Results\MobileNet_V2_Results.pkl'

with open(results_path, 'wb') as file:
    pickle.dump(MobileNet_V2_Results, file)

print(f"Results saved to {results_path}")


# ================= Loading the Training Results =================
# with open(results_path, 'rb') as file:
#     loaded_results = pickle.load(file)
#
# print(loaded_results)
