import torch
import random
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Fix random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# Image transformation class
class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

# Custom dataset class for hand gestures
class HandGestureDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.label_map = self._create_label_map()

    def _create_label_map(self):
        gesture_names = sorted({os.path.basename(os.path.dirname(p)) for p in self.file_list})
        return {name: idx for idx, name in enumerate(gesture_names)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.label_map[label_name]

        if self.transform:
            img = self.transform(img, self.phase)

        return img, label

# Function to make file list
def make_datapath_list(root="data", phase="train"):
    path_pattern = os.path.join(root, phase, "*", "*.jpg")
    return glob.glob(path_pattern, recursive=True)

# Function to create DataLoaders
def create_dataloaders(root="data", size=128, batch_size=32):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(size, mean, std)

    train_list = make_datapath_list(root=root, phase="train")
    val_list = make_datapath_list(root=root, phase="val")

    train_dataset = HandGestureDataset(train_list, transform, phase='train')
    val_dataset = HandGestureDataset(val_list, transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader}
