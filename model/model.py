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

class HandGestureModel(nn.Module):
    def __init__(self, base_model_name='mobilenet_v2', num_classes=14):
        super(HandGestureModel, self).__init__()

        if base_model_name == 'mobilenet_v2':
            self.base = models.mobilenet_v2(pretrained=True)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif base_model_name == 'vgg16':
            self.base = models.vgg16(pretrained=True)
            in_features = self.base.classifier[6].in_features
            self.base.classifier[6] = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError("Unsupported base model")

    def forward(self, x):
        return self.base(x)

# class model():
#     def __init__(self):
#         net1 = models.vgg16(pretrained=use_pretrained)
#         net2 = models.vgg19(pretrained=use_pretrained)
#         net3 = models.mobilnet(pretrained=use_pretrained)
        
#         net1.classifier[6] = nn.Linear(in_features=4096, out_features=16)
#         net2.classifier[6] = nn.Linear(in_features=4096, out_features=16)
#         net3.classifier[6] = nn.Linear(in_features=4096, out_features=16)

#     def forwordPass(self):


criterion = nn.CrossEntropyLoss()
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        # print("params_to_update_1:", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        #print("params_to_update_2:", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        #print("params_to_update_3:", name)

    else:
        param.requires_grad = False
        #print("no learning", name)

# Set learning rates
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)


# training function
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
