import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from constants import BATCH_SIZES
from dataset import FundusImageDataset

for batch_size in BATCH_SIZES:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dataloaders = {
    "train": DataLoader(FundusImageDataset(???), batch_size=batch_size)
  }

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.to(device)

print(model)

