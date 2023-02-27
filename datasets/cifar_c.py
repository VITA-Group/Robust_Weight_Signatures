import numpy as np
import os
import PIL
import torch
import torchvision
import random

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, root, name_list, train, transform=None):

        idxs = np.random.permutation(50000 if train else 10000)
        batch_idxs = np.array_split(idxs, len(name_list))

        for idx, name in enumerate(name_list):
            data_path = os.path.join(root, 'train' if train else 'test', name + '.npy')
            target_path = os.path.join(root, 'train' if train else 'test', 'labels.npy')
            
            if hasattr(self, 'data'):
                self.data[batch_idxs[idx]] = np.load(data_path)[batch_idxs[idx]]
                self.targets[batch_idxs[idx]] = np.int64(np.load(target_path))[batch_idxs[idx]]
            else:
                self.data = np.load(data_path)
                self.targets = np.int64(np.load(target_path))

        self.transform = transform
        
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        return self.transform(img), targets
    
    def __len__(self):
        return len(self.data)


def cifar10c_dataloaders(name, batch_size=128, data_dir='datasets/cifar10', num_workers=2):

    if not isinstance(name, list):
        name = [name]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    # print('Corruption Type: %s'%name)
    # print('Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t')
    # print('10000 images for testing\t no normalize applied in data_transform')
    # print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    idxs_train = sorted(random.sample(list(range(50000)), 45000))
    idxs_val = sorted(list(set(range(50000)) - set(idxs_train)))

    train_whole_set = CIFAR10C(data_dir, name, train=True, transform=train_transform)
    train_set = Subset(train_whole_set, idxs_train)
    val_set = Subset(train_whole_set, idxs_val)
    test_set = CIFAR10C(data_dir, name, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

    