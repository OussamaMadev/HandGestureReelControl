# === loadData.py ===
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class HG14Dataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.65, val_ratio=0.15):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_folder = os.path.join(data_dir, cls)
            all_imgs = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')])
            total = len(all_imgs)
            train_end = int(total * train_ratio)
            val_end = int(total * (train_ratio + val_ratio))

            if split == 'train':
                selected_imgs = all_imgs[:train_end]
            elif split == 'val':
                selected_imgs = all_imgs[train_end:val_end]
            else:
                selected_imgs = all_imgs[val_end:]

            for img in selected_imgs:
                self.image_paths.append(os.path.join(cls_folder, img))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.labels[idx]


def get_transforms(image_size=224):
    return {
        'train': A.Compose([
            A.Resize(height=image_size, width=image_size), 
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }


def create_dataloaders(data_dir, batch_size=32, image_size=224):
    transforms = get_transforms(image_size)

    loaders = {
        split: DataLoader(
            HG14Dataset(data_dir, transform=transforms['train' if split == 'train' else 'val'], split=split),
            batch_size=batch_size,
            shuffle=(split == 'train')
        )
        for split in ['train', 'val', 'test']
    }

    class_names = sorted(os.listdir(data_dir))
    return loaders, class_names
