import torch
from Config import *
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_transform = transforms.Compose([ transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

val_test_transform = transforms.Compose([ transforms.Resize((224,224)), transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def data_preparation(train_path, test_path, batch_size, val_size = 0.2):

    full_dataset = datasets.ImageFolder(root = train_path)
    train_size = int((1-val_size) * len(full_dataset))
    val_size = int(val_size * len(full_dataset))

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform

    test_dataset = datasets.ImageFolder(root = test_path, transform = val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader, test_loader


train_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Training'
test_path = 'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\Data\Testing'
train_loader, val_loader, test_loader = data_preparation(train_path, test_path, batch_size)
# Since we used random_split, access the original dataset using .dataset
class_names = train_loader.dataset.dataset.classes
