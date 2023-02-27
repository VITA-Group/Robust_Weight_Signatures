import os
import sys
import copy
import torch
import random
import argparse
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt 

from tools import *
from trainer import *
from datasets.cifar import cifar10_dataloaders, cifar100_dataloaders
from datasets.tiny_imagenet import tinyimagenet_dataloaders

def main(args):

    model = get_model(args.arch, pretrained=args.pretrained, path=args.pretrained_path, num_classes=args.num_classes).cuda()

    model, train_loader = setup_dataset(model, args)
    model.cuda()
    
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar10-c':
        _, _, std_test_loader = cifar10_dataloaders(batch_size = args.batch_size, 
            data_dir = args.std_data, num_workers = args.workers)
    elif 'cifar100' in args.dataset.lower():
        _, _, std_test_loader = cifar100_dataloaders(batch_size = args.batch_size, 
            data_dir = args.std_data, num_workers = args.workers)
    elif 'tinyimagenet' in args.dataset.lower():
        _, std_test_loader = tinyimagenet_dataloaders(batch_size = args.batch_size, 
            data_dir=args.std_data, num_workers=2)
    
    test_loaders, val_loaders = get_corruption_dataloaders(dataset=args.dataset, types=args.corruption, batch_size = args.batch_size, 
            data_dir = args.data, num_workers = args.workers)

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    all_result = {}
    all_result['train_acc'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []
    all_result['test_ra'] = []
    all_result['val_ra'] = []
    best_sa = 0
    best_ra = 0
    start_epoch = 0
    
    ################ start training #############

    for epoch in range(start_epoch, args.epochs):
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()

        # Evaluation
        clean_acc = test(std_test_loader, model, criterion, args)

        if len(args.corruption):
            racc_list = []
            for val_loader in val_loaders:
                racc = test(val_loader, model, criterion, args)
                racc_list.append(racc)
            racc = sum(racc_list) / len(val_loaders)

            racc_list = ['%.3f'%temp for temp in racc_list]

            test_racc_list = []
            for test_loader in test_loaders:
                test_racc = test(test_loader, model, criterion, args)
                test_racc_list.append('%.3f'%test_racc)
            print(len(racc_list))
            print_str = 'Corruption type %s: std test accuracy: %.3f, corruption accuracy: %s'%(
                ' '.join(args.corruption), clean_acc, ' '.join(racc_list if len(test_racc_list)==0 else test_racc_list))
        else:
            racc = 0
            print_str = 'std test accuracy: %.3f'%(clean_acc)
        
        print('EPOCH [%2d]: Training ACC %.2f, '%(epoch, train_acc) + print_str)

        is_sa_best = clean_acc > best_sa
        best_sa = max(clean_acc, best_sa)
        is_ra_best = racc > best_ra 
        best_ra = max(racc, best_ra)

        best_name_list = []
        if is_ra_best: best_name_list.append('model_RA_best.pth.tar')
        if is_sa_best: best_name_list.append('model_SA_best.pth.tar')
        checkpoint_state = {
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_checkpoint(checkpoint_state, args.save_dir, best_name=best_name_list)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning Benchmark")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

    parser.add_argument("--arch", default='vgg16')
    parser.add_argument("--dataset", default='cifar10-c')
    parser.add_argument('--data', type=str, help='location of the data corpus')
    parser.add_argument('--std_data', type=str)
    parser.add_argument('--corruption', nargs='+')
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--pretrained_path')

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='20,40', help='decreasing strategy')

    ##################################### Training Mode ###########################################
    # standard train

    parser.add_argument('--fedprox', action='store_true')
    parser.add_argument('--mu', type=float, default=0.01)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.corruption is None:
        args.corruption = []

    if args.dataset.lower() == 'cifar100' or args.dataset.lower() == 'cifar100-c':
        args.num_classes = 100
    elif args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar10-c':
        args.num_classes = 10
    elif args.dataset.lower() == 'tinyimagenet' or args.dataset.lower() == 'tinyimagenet-c':
        args.num_classes = 200

    print(args)

    main(args)

