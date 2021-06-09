# Transfer Learning 101: VGG16

When training a neural network using backpropagation, we update the weights of the network 
by following a gradient computed on an arbitrary loss function.
The gradient is (back)propagated using the chain rule of the derivative. As it is a product
of partial derivatives if these values are small, the gradient of the early layers tends towards zero.
Indeed, the product $0.125 * 0.25 * 0.5 = 0.02$ now imagine a product of thousands of weights < 1.
This is also know as the *vanishing gradient*. A wise choice of activation function, and a large number of
data help to counteract this effect.

![image](vanishing-gradient.ppm =600x300)

Transfer learning is taking the advantage of the *vanishing gradient*. The idea is to chop the first layers
from a pretrained model and replace the last section with a custom output network, to better fit the needs of our
specific task. As the first layers of the previously trained model are frozen, they won't vary significantly during
our training. Even if the pretrained model was built to solve a completely different problem.

Training a complete neural network is expensive, the possibility to use pre-fitted models is one of the big driver of machine
learning's commercial success as they require less training time, less engineering and less training data.

### Table of content
${toc}

### VGG16

VGG16 is a convolutional neural network from a 2014 paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
I won't go into detail about what a CNN is but as a condensed remainder it is a type of neural network largely used in computer vision topics. The input of a
CNN is an "image" or a matrix of scalars, which means that the information contained in an image is in 2 dimension.
The CNN filters 2d pattern by using smaller matrices known as *kernels*. Thanks to these filters the network better
catches 2d patterns contained in images.
VGG16 was originally trained on the [ImageNet Challenge Dataset](http://www.image-net.org/challenges/LSVRC/) which is a multi-classification challenge.
The ImageNet's dataset contains images from 1000 different classes. Eg: dogs, sailboats, trimarans, huskies,... 

### The task

We will use the trained VGG16 model from the 2014 paper on the [Tomato leaf disease detection](https://www.kaggle.com/kaustubhb999/tomatoleaf) dataset.
As we discussed in the introduction, the new task is completely different from the classification the model was built to make.

#### Loading the data & data augmentation

We load the data using `torchvision`'s `ImageFolder` class, which is similar to a `flow_from_directory` generator from `keras`.
```python
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(
		'train',
		transform=train_transform,
	)

test_dataset = ImageFolder(
		'val',
		transform=test_transform,
	)
```

Data augmentation artificially increases the size of the training set by generating many variants of each training instance.
For example, for each image in the training set we generate 20 new images by translating/fliping/rescaling randomly the original
image.

```python
import torchvision
import PIL

train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor()
    ])

test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
```

#### Customizing VGG16

```python
from torch import nn

# Get the pretrained model
model = torchvision.models.vgg16(pretrained=True)

# Freeze its parameters
for param in model.features.parameters():
    param.requires_grad = False

#Adapt the classifier
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=10, bias=True)
        )
```

#### Training

As the dataset is split in a train and a test set and the class are well balanced we will use a simple cross validation and
an unweighted cross entropy loss. I used an Adam optimizer for the gradient descent.

```python
import torch
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-5


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=64,
		suffle=True
	)

test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=64
	)

max_acc = 0

for epoch
```
