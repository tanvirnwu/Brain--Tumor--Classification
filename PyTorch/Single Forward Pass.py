from MyModels import *
from DataProcessing import *
from config import *
from torchinfo import summary

train_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Training'
test_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Testing'
train_loader, val_loader, test_loader = data_preparation(train_path, test_path, batch_size)


def single_forward_pass(model_instance: nn.Module):
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_loader))
    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")
    model_instance.to(device)

    # Set the model to evaluation mode
    model_instance.eval()

    # Perform a forward pass on the single image
    with torch.inference_mode():
        pred = model_instance(img_single.to(device))

    # Print out the results
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

    # Sumamry of the model
    #summary(model_instance, input_size=[1, 3, 224, 224])



# Instantiate the model
model_instance = MyModel_1(input_shape=3, hidden_units=32, output_shape=4)
single_forward_pass(model_instance)
