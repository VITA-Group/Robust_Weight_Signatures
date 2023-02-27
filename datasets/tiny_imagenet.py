import os
import sys
import torch
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

def tinyimagenet_c_trainloaders(name, serverity, batch_size, data_dir, num_workers=2):

    if not isinstance(name, list):
        name = [name]

    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_idxs = np.array_split(np.random.permutation(100000), len(name))

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train', name[0], f'{serverity}'), train_transform)
    
    if len(name) > 1:
        for idx, temp_name in enumerate(name[1:]):
            temp_train_set = datasets.ImageFolder(os.path.join(data_dir, 'train', temp_name, f'{serverity}'), train_transform)
            idxs = batch_idxs[idx+1].tolist()
            for sample_idx in idxs:
                train_set.samples[sample_idx] = temp_train_set.samples[sample_idx]
    train_set.imgs = train_set.samples
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader

def tinyimagenet_c_testloaders(name, serverity, batch_size, data_dir, num_workers=2, shuffle=True):

    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = datasets.ImageFolder(os.path.join(data_dir, 'val', name, f'{serverity}'), test_transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return test_loader

def tinyimagenet_dataloaders(batch_size, data_dir, num_workers=2, shuffle=True):

    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
    
