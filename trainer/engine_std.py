import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from trainer.utils import * 

__all__ = ['train_epoch', 'test']


def train_epoch(train_loader, model, criterion, optimizer, epoch, args, pretrain_state_dict=None):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        if args.fedprox:
            fed_prox_reg = 0.0
            for param_name, param in model.named_parameters():
                fed_prox_reg += ((args.mu / 2) * torch.norm((param - pretrain_state_dict[param_name]))**2)
            loss += fed_prox_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i == len(train_loader):
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    # print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    #     if i % args.print_freq == 0:
    #         end = time.time()
    #         print('Test: [{0}/{1}]\t'
    #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #             'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
    #             'Time {2:.2f}'.format(
    #                 i, len(val_loader), end-start, loss=losses, top1=top1))
    #         start = time.time()

    # print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg