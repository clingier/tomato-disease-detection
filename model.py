import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from dataset import train_dataset, test_dataset
from tqdm import tqdm
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64
NCLASSES = 10
LEARNING_RATE = 1e-5
NEPOCHS = 10
PATH = f"model_{datetime.now()}.pt"

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Writer History for Tensorboard integration
writer = SummaryWriter()

# Get the pretrained model
model = torchvision.models.vgg16(pretrained=True)

# Freeze its parameters
for param in model.features.parameters():
    param.requires_grad = False

#Adapt the classifier
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=512, out_features=10, bias=True)
        )
model = model.to(device=device)

#Loss function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Train and Test DataLoader
train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
        )

#Check model accuracy on a given loader
def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            
            _, y_pred = scores.max(1)
            num_correct += (y_pred == y).sum()
            num_samples += y_pred.size(0)

    print(f"Accuracy is {num_correct/num_samples*100:.2f}%")
    return num_correct / num_samples

#Training NEPOCHS
max_acc = 0

for epoch in range(NEPOCHS):
    losses = []

    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = check_accuracy(test_loader, model)
    writer.add_scalar('Loss/train', loss, epoch) 
    writer.add_scalar('Accuracy/test', acc, epoch)

    if acc > max_acc:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, PATH)
        max_acc = acc
