import glob
import os.path as osp
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms

use_pretrained = True
from torchvision.models import mobilenet_v2, vgg16, efficientnet_b0

class HandGestureModel(nn.Module):
    def __init__(self, base_model_name='mobilenet_v2', num_classes=14, pretrained=True, weight_path=None):
        super(HandGestureModel, self).__init__()

        if base_model_name == 'mobilenet_v2':
            self.base = mobilenet_v2(pretrained=pretrained)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        elif base_model_name == 'vgg16':
            self.base = vgg16(pretrained=pretrained)
            in_features = self.base.classifier[6].in_features
            self.base.classifier[6] = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        elif base_model_name == 'efficientnet_b0':
            self.base = efficientnet_b0(pretrained=False)  # Load custom weights later
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            if weight_path is not None:
                state_dict = torch.load(weight_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"Loaded EfficientNet weights from {weight_path}")

        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

    def forward(self, x):
        return self.base(x)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    accuracy_list = []
    loss_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device：", device)
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # epoch loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders_dict[phase]):

                print (f'Iteration: ', i)
                print (labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # Calcurate loss
                    _, preds = torch.max(outputs, 1)  # Prediction

                    # Back propagtion
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # update loss summation
                    epoch_loss += loss.item() * inputs.size(0)
                    # update correct prediction summation
                    epoch_corrects += torch.sum(preds == labels.data)

            # loss and accuracy for each epoch loop
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                accuracy_list.append(epoch_acc.item())
                loss_list.append(epoch_loss)

    return accuracy_list, loss_list

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return loss / total, correct / total
