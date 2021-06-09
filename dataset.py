from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Resize, RandomHorizontalFlip, RandomRotation, ToTensor
from torchvision.transforms.functional import adjust_saturation
import torch
from torch import nn
import torchvision
import PIL

# Preprocessing and Data Augmentation

train_transform = torchvision.transforms.Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(20, resample=PIL.Image.BILINEAR),
        ToTensor(), 
        Lambda(lambda img: adjust_saturation(img, saturation_factor=3))
    ])

test_transform = torchvision.transforms.Compose([
        Resize((224, 224)),
        ToTensor(),
        Lambda(lambda img: adjust_saturation(img, saturation_factor=3))
    ])

# Datasets
train_dataset = ImageFolder('train',
        transform=train_transform,
    )

test_dataset = ImageFolder(
            'val', 
            transform=test_transform,
        )
