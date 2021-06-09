import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision

from dataset import test_dataset

writer = SummaryWriter()

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = torch.load('model_2021-01-27 14:53:01.179116.pt')

images, labels = next(iter(test_loader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.close()
