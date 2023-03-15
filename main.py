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





# The test transform is meant to be applied just to the very final test images.
test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([150, 150]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Preprocessing the data and creating the DataLoaders.
images_dir = torchvision.datasets.ImageFolder("data", transform=train_transform)
len_data = len(images_dir)
image_train, test_raw = torch.utils.data.random_split(images_dir, (len_data - 2000, 2000))

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

# We need to re-organize the test data becauee of the random split and the dataset itself
## The images are not organized in classes.

def re_organize_test_folder(test_imgs):
    matched_pair = list()

    for idx in range(len(images_dir.classes)):
        matched = list()
        for image in test_raw:
            print(image[1], idx)
            if image[1] == idx:
                matched.append(image)
            else:
                pass
        matched_pair.append(matched)

    res = []
    for pair in matched_pair:
        for sample in pair:
            res.append(sample)

    return res


image_test = re_organize_test_folder(test_raw)

train_loader = torch.utils.data.DataLoader(image_train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(image_test, batch_size=64, shuffle=False)


# First Module -- derived from the exercise seen in class for MINST classification.


class first_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 10, 3)

        # Slightly re-arranged the dimensionality of the hidden layers to fit this problem.
        self.fc1 = nn.Linear(213160, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 19)
        self.relu = nn.ReLU()

    def forward(self, x):
        # first convolution
        x = self.conv1(x)
        x = self.relu(x)

        # second convolution
        x = self.conv2(x)
        x = self.relu(x)

        # fully connected
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch

        # fc1
        x = self.fc1(x)
        x = self.relu(x)

        # fc2
        x = self.fc2(x)
        x = self.relu(x)

        # fc out
        x = self.fc3(x)

        return x


# Second module, we added Dropout to reduce overfitting and the MaxPooling to reduce dimensionality


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


# Here it is the checkpoint for the second model, I couldn't provide the graph as
## I was working for the first time with Kaggle Notebooks (Google Colab did not let me use GPU anymore)
### And it doesn't save the versions of the notebook automatically (as colab does). So I lost the losses info, but luckily I have saved the checkpoint.
# Loading the model with the cpu as a device because I do not have cuda cores in my laptop.

model = torch.load('mymodel_cpu.pth')
model.eval()

print(model)

''' Here we just change the dense layer of the ResNet50 to fit out problem.
    It is not shown the training of this model here, but if needed I will show some training examples
    on google colab. It generally achieves better results than my model being pre-trained on tons of images.'''

model_with_resnet = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
lastLayer = model_with_resnet.fc.in_features

model_with_resnet.fc = nn.Sequential(nn.Linear(2048, 256),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(256, 19))

'''-----------------------------------------'''

''' For the VGG16, I have found online an implementation of the model from scratch, which is
something different from one could expect from torchvision's models. Here is the Network Architecture.'''


## In this case, we should modify the shape of the images to be 224x224. It could be easily done changing the
## tranformation of the ImageFolder instantiation.

class VGG16(nn.Module):
    def __init__(self, num_classes=19):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


''' As we can see, it is a huge network, which means that for my hardware it will be difficult to train
this one in a reasonable amount of time. But it does perform well (at least in the first steps)... I did
observe that in order to make it run I had to reduce the batch size down to 16 images per step.'''

''' Now it is time to Train the model(s). Here you can still find the weights and input type as cuda 
    since I was working with the notebook to train the models.'''


def trainingLoop(train_dataloader, model, loss_fn, optimizer):
    train_loss = 0

    model.train()  # Put the model in training mode
    for batch, (X, y) in enumerate(train_dataloader):
        # move data on gpu
        X, y = X.cuda(), y.cuda()

        pred = model(y)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"The loss is {train_loss / len(X.dataset)}")


def testLoop(test_dataloader, model, loss_fn):
    print_size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.cuda(), y.cuda()

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss = test_loss / num_batches
    correct = correct / print_size

    print(f"Accuracy: {correct * 100}, Average loss: {test_loss}")


epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for e in range(epochs):
    trainingLoop(train_loader, model, loss_fn, optimizer)
    testLoop(test_loader, model, loss_fn)



'''Testing the model (2nd model) with some picture taken on the internet'''

test_paths = ["tests/coral1.jpg", "tests/crab1.jpg", "tests/nudi1.jpg", "tests/seal1.jpg"]
img_testing = [Image.open(test_img) for test_img in test_paths]

# Stacking and applying all the transformation except the randomflip to the test images
test_batch = torch.stack([test_transform(sample) for sample in img_testing])

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


