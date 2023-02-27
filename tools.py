import os
import torch
from models.vgg import vgg16
from models.resnet import resnet50
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from advertorch.utils import NormalizeByChannelMeanStd

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from trainer import *
from datasets.cifar_c import cifar10c_dataloaders
from datasets.cifar import cifar10_dataloaders, cifar100_dataloaders

corruption_types_all = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
    'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
    'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loaders, model, corruption):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for corrupt in corruption:
        loader = loaders[corrupt]
        for input, _ in loader:
            input = input.cuda()
            b = input.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def get_projection(a, param):
    assert len(a.shape) == 1
    a = a.reshape(1, -1)
    try:
        projection_matrix = (a.T.matmul(a)) / (a.matmul(a.T))
        projections = projection_matrix.matmul(param.T).T
        return param - projections
    except:
        a = a / torch.norm(a)
        norm = torch.norm(param, dim=1).mul(torch.cosine_similarity(param, a, dim=1))
        projections = torch.diag(norm).matmul(a.repeat(len(param),1)) 
        return param - projections

def get_dirs(root_dir, corruption):

    dirs = {}
    for temp_dir in os.listdir(root_dir):
        for corruption_type in corruption:
            if corruption_type in temp_dir:
                dirs[corruption_type] = os.path.join(root_dir, temp_dir, 'model_RA_best.pth.tar')
                break

    return dirs

        
def get_attack_strength(args, eps, norm):
    num_step = 10
    if norm == 'linf':
        args.test_eps, args.test_alpha, args.test_step, args.test_norm = eps/255, 2.5*eps/num_step/255, num_step, 'l_inf'
    elif norm == 'l2':
        args.test_eps, args.test_alpha, args.test_step, args.test_norm = eps, 2.5*eps/num_step, num_step, 'l_2'
    return args

def get_model(arch, pretrained, num_classes, path=None):
    if arch.lower() == 'vgg16':
        model = vgg16(pretrained=False, progress=True, num_classes=num_classes)

    elif arch.lower() == 'resnet50':
        model = resnet50(pretrained=False, progress=True, num_classes=num_classes)
    
    else:
        assert False
    
    if pretrained:
        state_dict = torch.load(path)
        if arch.lower() == 'resnet50':
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        keys_source = set(state_dict.keys())
        keys_target = set(model.state_dict().keys())
        print('Contained in pretrained model but not loaded in the target model: ', keys_source-keys_target)
        print('Contained in the target model but be loaded: ', keys_target-keys_source)
        model.load_state_dict(state_dict, strict=False)

    return model

def setup_dataset(model, args):
    if args.dataset.lower() == 'cifar10':
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        from datasets.cifar import cifar10_dataloaders
        train_loader, _, _ = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset.lower() == 'cifar100':
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        from datasets.cifar import cifar100_dataloaders
        train_loader, _, _ = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    elif args.dataset.lower() == 'cifar10-c' or args.dataset.lower() == 'cifar100-c':
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        from datasets.cifar_c import cifar10c_dataloaders
        train_loader, _, _ = cifar10c_dataloaders(name=args.corruption, batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    elif args.dataset.lower() == 'tinyimagenet':
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        from datasets.tiny_imagenet import tinyimagenet_dataloaders
        train_loader, _ = tinyimagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers=2)
    
    elif args.dataset.lower() == 'tinyimagenet-c':
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        from datasets.tiny_imagenet import tinyimagenet_c_trainloaders
        train_loader = tinyimagenet_c_trainloaders(name=args.corruption, serverity=5, batch_size = args.batch_size, data_dir = args.data, num_workers=2)

    else:
        raise ValueError('Dataset not supprot yet !')

    model.normalize = normalization
    print(model)

    return model, train_loader

def get_corruption_dataloaders(dataset, types, batch_size, data_dir, num_workers):
    test_loaders, val_loaders = [], []
    if 'cifar' in dataset.lower():
        for name in types:
            # cifar10c_dataloaders also support cifar100
            _, temp_val_loader, temp_test_loader = cifar10c_dataloaders(name=name, batch_size = batch_size, 
                data_dir = data_dir, num_workers = num_workers)

            val_loaders.append(temp_val_loader)
            test_loaders.append(temp_test_loader)
    
    elif 'tinyimagenet' in dataset.lower():
        for name in types:
            from datasets.tiny_imagenet import tinyimagenet_c_testloaders
            temp_val_loader = tinyimagenet_c_testloaders(name=name, serverity=5, 
                batch_size=batch_size, data_dir=data_dir, num_workers=num_workers)

            val_loaders.append(temp_val_loader)

    return test_loaders, val_loaders

def test_all(model, test_loader, corruption_data_loaders, criterion, names, args):

    result = {}
    result['clean'] = test(test_loader, model, criterion, args)

    for name in names:
        if 'pgd' in name:
            attack_type = 'linf' if 'linf' in name else 'l2'
            args = get_attack_strength(args, float(name.split('_')[1].strip('eps')), attack_type)
            result[name] = test_adv(test_loader, model, criterion, args)
        elif name in corruption_types_all:
            result[name] = test(corruption_data_loaders[name], model, criterion, args)
        else:
            assert False
    return result

def select_layers(state_dict, keep_num, arch='vgg16'):
    num = 0
    key_list = []
    for key in state_dict.keys():
        if arch == 'vgg16':
            if len(state_dict[key].shape) == 4:
                num +=1
                key_list.append('features.'+key.split('.')[1]+'.weight')
                key_list.append('features.'+key.split('.')[1]+'.bias')

                if num == keep_num:
                    break

        elif arch == 'resnet50':
            if 'layer' in key and 'conv1' in key:
                num += 1
                identifier = key[:9]
                key_list.append(identifier+'conv1.weight')
                key_list.append(identifier+'bn1.weight')
                key_list.append(identifier+'bn1.bias')
                # key_list.append(identifier+'bn1.running_mean')
                # key_list.append(identifier+'bn1.running_var')
                key_list.append(identifier+'conv2.weight')
                key_list.append(identifier+'bn2.weight')
                key_list.append(identifier+'bn2.bias')
                # key_list.append(identifier+'bn2.running_mean')
                # key_list.append(identifier+'bn2.running_var')
                key_list.append(identifier+'conv3.weight')
                key_list.append(identifier+'bn3.weight')
                key_list.append(identifier+'bn3.bias')
                # key_list.append(identifier+'bn3.running_mean')
                # key_list.append(identifier+'bn3.running_var')
                key_list.append(identifier+'downsample.0.weight')
                key_list.append(identifier+'downsample.1.weight')
                key_list.append(identifier+'downsample.1.bias')
                # key_list.append(identifier+'downsample.1.running_mean')
                # key_list.append(identifier+'downsample.1.running_var')
                
            elif 'conv1' in key:
                num += 1
                key_list.append('conv1.weight')
                key_list.append('bn1.weight')
                key_list.append('bn1.bias')
                # key_list.append('bn1.running_mean')
                # key_list.append('bn1.running_var')
            
            if num == keep_num:
                    break

    return key_list

