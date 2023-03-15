import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.transforms import transforms



class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # I've used Kernel of size 3x3 all the time.

        # -- Convolutional Layers -- #
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)

        # -- Dense Layers -- #
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, 19)

        # -- Activation functions and others -- #
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        ## Here there is the maxpooling with a kernel of size (2x2) and a stride of 2x2
        self.maxpooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpooling(x)

        x = self.relu(self.conv2(x))
        x = self.maxpooling(x)

        x = self.relu(self.conv3(x))
        x = self.maxpooling(x)

        x = self.relu(self.conv4(x))
        x = self.maxpooling(x)

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        out = self.fc2(x)
        return out

test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

test_paths = ["tests/coral1.jpg", "tests/crab1.jpg", "tests/nudi1.jpg", "tests/seal1.jpg"]
img_testing = [Image.open(test_img) for test_img in test_paths]

# Stacking and applying all the transformation except the randomflip to the test images
test_batch = torch.stack([test_transform(sample) for sample in img_testing])


model = torch.load('mymodel.pth', map_location=torch.device('cpu'))
class_names = ['Corals',
               'Crabs',
               'Dolphin',
               'Eel',
               'Jelly Fish',
               'Lobster',
               'Nudibranchs',
               'Octopus',
               'Penguin',
               'Puffers',
               'Sea Rays',
               'Sea Urchins',
               'Seahorse',
               'Seal',
               'Sharks',
               'Squid',
               'Starfish',
               'Turtle_Tortoise',
               'Whale']

with torch.no_grad():
    model.eval()
    out = model(test_batch)
    out = nn.functional.softmax(out, dim=1)

preds = torch.max(out, 1).indices

b = 0
for i, img in enumerate(img_testing):
    plt.title(f"Prediction: {class_names[preds[b]]} ]")
    plt.imshow(img)
    plt.show()
    b += 1

