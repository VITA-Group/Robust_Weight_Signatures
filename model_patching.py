import os
import sys
import torch
import random
import argparse
import itertools
import numpy as np 
import pandas as pd
import torch.nn as nn 
import matplotlib.pyplot as plt 

from tools import *
from trainer import *
from quantize_utils.func import linear_quant
from datasets.cifar_c import cifar10c_dataloaders
from datasets.cifar import cifar10_dataloaders, cifar100_dataloaders
from datasets.tiny_imagenet import tinyimagenet_c_testloaders, tinyimagenet_dataloaders, tinyimagenet_c_trainloaders

from advertorch.utils import NormalizeByChannelMeanStd

default_data_dir_tinyimagenet = ''

def decompose(state_dicts, pretrained_weight, patched_weight, base_weight, weight=None, only_backbone=True, keep_num=13, 
    quantization=None, quantization_method='linear'):

    key_list = select_layers(state_dicts[0], keep_num, args.arch)

    new_dict = {}
    if not only_backbone:
        assert False
    
    else:
        for key in state_dicts[0].keys():

            if ('bn' in key or 'downsample' in key) and 'num' not in key:
                new_dict[key] = torch.mean(torch.stack([state[key] for state in state_dicts], dim=0), dim=0)
            
            elif key in key_list:
                param = torch.stack(
                    [(state[key]-pretrained_weight[key].cuda()) * weight[idx] for idx, state in enumerate(state_dicts)], dim=0)
                param = param.reshape(len(state_dicts), -1)
                vectors = get_projection((base_weight[key]-pretrained_weight[key].cuda()).flatten(), param)
            
                # projections = torch.mean(projections, dim=0)

                if quantization:
                    if quantization_method == 'log':
                        sign = torch.sign(vectors)
                        vectors = torch.log2(torch.abs(vectors))
                        linear_quant(vectors, quantization>>1, min=torch.min(vectors).detach().item(), max=torch.max(vectors).detach().item(), 
                            clamp=True, inplace=True)
                        vectors = sign * torch.exp2(vectors)

                    elif quantization_method == 'linear':
                        vectors = linear_quant(vectors, quantization, min=torch.min(vectors).detach().item(), max=torch.max(vectors).detach().item(), 
                            clamp=False, inplace=False)
                    
                    elif quantization_method == 'tanh':
                        vectors = torch.tanh(vectors)
                        vectors = linear_quant(vectors, quantization, min=torch.min(vectors).detach().item(), max=torch.max(vectors).detach().item(), 
                            clamp=False, inplace=False)
                        vectors = torch.atanh(vectors)

                vector_combinations = torch.sum(torch.diag(weight).matmul(vectors), dim=0)
                new_dict[key] = vector_combinations.reshape_as(pretrained_weight[key]) + patched_weight[key]
                del vector_combinations, vectors, param
            
            else:
                # print(f'keep {key} unchanged')
                new_dict[key] = patched_weight[key]
                        
        return new_dict


def main(args):

    dirs = get_dirs(args.corruption_model_root, args.corruption)
    criterion = nn.CrossEntropyLoss()

    # get dataloaders
    corruption_data_loaders = {}
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        for name in corruption_types_all:
            _, _, temp_test_loader = cifar10c_dataloaders(name, batch_size=128, data_dir=args.corruption_data, num_workers=2)
            corruption_data_loaders[name] = temp_test_loader
    elif args.dataset == 'tinyimagenet':
        for name in corruption_types_all:
            temp_test_loader = tinyimagenet_c_testloaders(name=name, serverity=args.serverity, batch_size=128, data_dir=args.corruption_data, num_workers=2)
            corruption_data_loaders[name] = temp_test_loader
            
    
    if args.dataset == 'cifar10':
        _, _, test_loader = cifar10_dataloaders(batch_size=128, data_dir=args.data, num_workers=2)
        normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif args.dataset == 'cifar100':
        _, _, test_loader = cifar100_dataloaders(batch_size=128, data_dir=args.data, num_workers=2)
        normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif args.dataset == 'tinyimagenet':
        _, test_loader = tinyimagenet_dataloaders(batch_size=64, data_dir=args.data, num_workers=2)
        normalization = NormalizeByChannelMeanStd(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])

    # get base model
    model = get_model(args.arch, pretrained=False, num_classes=args.num_classes)
    
    model.normalize = normalization
    model.cuda()
    patched_weight = torch.load(args.patched_model)['state_dict']
    model.load_state_dict(patched_weight, strict=True)
    
    std_result = test_all(model, test_loader, corruption_data_loaders, criterion, args.corruption, args)
    print(std_result)
    
    # combine vectors
    pretrained_weight = torch.load(args.pretrained)
    state_dicts = []
    for dir_temp in dirs.values():
        state_dicts.append(torch.load(dir_temp)['state_dict'])
    
    base_weight = torch.load(args.base_model)['state_dict']
    
    if args.alpha:
        weight = torch.tensor([1] * len(state_dicts)).cuda().float()
        weight = args.alpha * weight / len(state_dicts)
    elif args.finetune_alpha:
        weight = torch.tensor(args.finetune_alpha).cuda()

    new_dict = decompose(state_dicts, pretrained_weight, model.state_dict(), base_weight,
        weight=weight, only_backbone=True, keep_num=args.keep_num, quantization=args.quantization, quantization_method=args.quantization_type)
    model.load_state_dict(new_dict)

    if 'resnet' in args.arch:
        assert args.dataset == 'tinyimagenet'
        corruption_train_loaders = {}
        for name in args.corruption:
            temp_train_loader = tinyimagenet_c_trainloaders(name=name, serverity=5, batch_size=128, data_dir=default_data_dir_tinyimagenet, num_workers=2)
            corruption_train_loaders[name] = temp_train_loader
        bn_update(corruption_train_loaders, model, args.corruption)
        new_bn_dict = model.state_dict()
    
        for key in new_dict:
            if 'running_mean' in key or 'running_var' in key:
                if args.alpha < 1:
                    new_dict[key] = (1-args.alpha) * new_dict[key] + new_bn_dict[key] * args.alpha
                else:
                    new_dict[key] = new_bn_dict[key]
        model.load_state_dict(new_dict)

    patch_result = test_all(model, test_loader, corruption_data_loaders, criterion, 
            corruption_types_all if args.test_all_corruptions else args.corruption, args)
    print(patch_result)

    if os.path.isfile(args.save_log):
        save_log = torch.load(args.save_log)
    else:
        save_log = {'patched': {'clean': [], 'robust': {}}, 'std': {'clean': [], 'robust': {}}}
    save_log['std']['clean'].append(std_result['clean'])
    save_log['std']['robust'][args.corruption[0]] = std_result[args.corruption[0]]
    save_log['patched']['clean'].append(patch_result['clean'])
    save_log['patched']['robust'][args.corruption[0]] = patch_result[args.corruption[0]]

    torch.save(save_log, args.save_log)
    print('Standard MODEL: avg TA %.2f, avg RA %.2f'%(torch.mean(torch.tensor(save_log['std']['clean'])), 
        torch.mean(torch.tensor(list(save_log['std']['robust'].values())))))
    print('Patched  MODEL: avg TA %.2f, avg RA %.2f'%(torch.mean(torch.tensor(save_log['patched']['clean'])), 
        torch.mean(torch.tensor(list(save_log['patched']['robust'].values())))))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning Benchmark")
    parser.add_argument('--corruption_model_root')
    parser.add_argument('--corruption', nargs='+')

    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument("--arch", default='vgg16')
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument('--data', type=str)
    parser.add_argument('--corruption_data', type=str)
    parser.add_argument('--serverity', default=5, type=int)

    parser.add_argument('--base_model')
    parser.add_argument('--patched_model')
    parser.add_argument('--pretrained')
    parser.add_argument('--save_log')

    parser.add_argument('--keep_num', default=7, type=int)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--finetune_alpha', nargs='+', type=float)
    parser.add_argument('--quantization', type=int, default=None)
    parser.add_argument('--quantization_type', type=str)

    parser.add_argument('--test_all_corruptions', action='store_true')
    args = parser.parse_args()

    if args.dataset.lower() == 'cifar10':
        args.num_classes = 10
    elif args.dataset.lower() == 'cifar100':
        args.num_classes = 100
    elif 'tinyimagenet' in args.dataset.lower():
        args.num_classes = 200
    print(args.keep_num, args.alpha, args.finetune_alpha, args.corruption)

    if not args.patched_model:
        args.patched_model = args.base_model
    
    main(args)