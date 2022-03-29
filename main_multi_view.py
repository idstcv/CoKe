#!/usr/bin/env python
# Copyright (c) Alibaba Group
import argparse
import builtins
import os
import random
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import coke.loader
import coke.folder
import coke.optimizer
import coke.builder_multi_view
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 64)')
parser.add_argument('--epochs', default=801, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log', type=str)
# options for coke
parser.add_argument('--coke-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--coke-num-ins', default=1281167, type=int,
                    help='number of instances (default: 1281167)')
parser.add_argument('--coke-num-head', default=3, type=int,
                    help='number of k-means ( default: 3)')
parser.add_argument('--coke-k', default=[3000, 4000, 5000], type=int, nargs="+", help='multi-clustering head')
parser.add_argument('--coke-t', default=0.05, type=float,
                    help='temperature (default: 0.05)')
parser.add_argument('--coke-dual-lr', default=20., type=float,
                    help='dual learning rate')
parser.add_argument('--coke-ratio', default=0.4, type=float,
                    help='ratio of lower-bound')
parser.add_argument('--coke-alpha', default=0.2, type=float,
                    help='weight of one-hot label')
parser.add_argument('--coke-beta', default=0.5, type=float,
                    help='weight of regular views')
parser.add_argument('--coke-snum', default=6, type=int,
                    help='number of small views')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    assert (args.coke_num_head == len(args.coke_k))
    print("=> creating model '{}'".format(args.arch))
    model = coke.builder_multi_view.CoKe(
        base_encoder=models.__dict__[args.arch],
        K=args.coke_k,
        dim=args.coke_dim,
        num_ins=args.coke_num_ins,
        num_head=args.coke_num_head,
        T=args.coke_t,
        dual_lr=args.coke_dual_lr,
        ratio=args.coke_ratio
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    model.module.load_param()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = coke.optimizer.LARS(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # augmentation for two regular views
    aug_1 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([coke.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    aug_2 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([coke.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([coke.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    # augmentation for small views
    aug_s = [
        transforms.RandomResizedCrop(96, scale=(0.05, 0.6)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([coke.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = coke.folder.ImageFolder(
        traindir,
        coke.loader.MultiCropsTransform(transforms.Compose(aug_1),
                                        transforms.Compose(aug_2), transforms.Compose(aug_s),
                                        args.coke_snum))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, scaler)
        model.module.update_center()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='model/{}_{:04d}.pth.tar'.format(args.log, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_large = AverageMeter('Loss_large', ':.4e')
    losses_small = AverageMeter('Loss_small', ':.4e')
    a_top1 = AverageMeter('aAcc@1', ':6.2f')
    a_top5 = AverageMeter('aAcc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, a_top1, a_top5, losses, losses_large, losses_small],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args, i, train_loader_len)
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            for j in range(0, args.coke_snum):
                images[2][j] = images[2][j].cuda(args.gpu, non_blocking=True)
        # compute output
        target = target.cuda(args.gpu)
        with autocast():
            pred_view1, pred_view2, proj_view1, proj_view2, pred_s, proj_s, labels = model(images[0], images[1],
                                                                                           images[2], target, epoch)
            loss_pred = 0
            loss_proj = 0
            loss_pred_small = 0
            loss_proj_small = 0
            with torch.no_grad():
                soft_label_view1 = []
                soft_label_view2 = []
                for j in range(0, args.coke_num_head):
                    soft_label_view1.append(F.softmax(proj_view1[j], dim=1))
                    soft_label_view2.append(F.softmax(proj_view2[j], dim=1))
            # compute loss on 2x224 crops with 3 heads
            for j in range(0, args.coke_num_head):
                loss_pred += args.coke_alpha * criterion(pred_view1[j], labels[j])
                loss_pred += args.coke_alpha * criterion(pred_view2[j], labels[j])
                loss_proj += args.coke_alpha * criterion(proj_view1[j], labels[j])
                loss_proj += args.coke_alpha * criterion(proj_view2[j], labels[j])
                loss_pred -= (1. - args.coke_alpha) * torch.mean(
                    torch.sum(F.log_softmax(pred_view1[j], dim=1) * soft_label_view2[j], dim=1))
                loss_pred -= (1. - args.coke_alpha) * torch.mean(
                    torch.sum(F.log_softmax(pred_view2[j], dim=1) * soft_label_view1[j], dim=1))
                loss_proj -= (1. - args.coke_alpha) * torch.mean(
                    torch.sum(F.log_softmax(proj_view1[j], dim=1) * soft_label_view2[j], dim=1))
                loss_proj -= (1. - args.coke_alpha) * torch.mean(
                    torch.sum(F.log_softmax(proj_view2[j], dim=1) * soft_label_view1[j], dim=1))
                # compute loss on 6 small views with 3 heads
                for k in range(0, args.coke_snum):
                    loss_pred_small += 2. * args.coke_alpha * criterion(pred_s[j * args.coke_snum + k], labels[j])
                    loss_pred_small -= (1. - args.coke_alpha) * torch.mean(torch.sum(
                        F.log_softmax(pred_s[j * args.coke_snum + k], dim=1) * (
                                soft_label_view1[j] + soft_label_view2[j]), dim=1))
                    loss_proj_small += 2. * args.coke_alpha * criterion(proj_s[j * args.coke_snum + k], labels[j])
                    loss_proj_small -= (1. - args.coke_alpha) * torch.mean(torch.sum(
                        F.log_softmax(proj_s[j * args.coke_snum + k], dim=1) * (
                                soft_label_view1[j] + soft_label_view2[j]), dim=1))
            # compute averaged loss
            loss_large = (0.5 / (2. * args.coke_num_head)) * (loss_pred + loss_proj)
            loss_small = (0.5 / (2. * args.coke_snum * args.coke_num_head)) * (loss_pred_small + loss_proj_small)
            loss = args.coke_beta * loss_large + (1. - args.coke_beta) * loss_small
        a_acc1, a_acc5 = accuracy(pred_view1[0], labels[0], topk=(1, 5))
        a_top1.update(a_acc1[0], images[0].size(0))
        a_top5.update(a_acc5[0], images[0].size(0))
        losses.update(loss.item(), images[0].size(0))
        losses_large.update(loss_large.item(), images[0].size(0))
        losses_small.update(loss_small.item(), images[0].size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(train_loader_len)
    for i in range(0, args.coke_num_head):
        print('max and min cluster size for {}-class clustering is ({},{})'.format(args.coke_k[i], torch.max(
            model.module.counters[i].data).item(), torch.min(model.module.counters[i].data).item()))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (state['epoch'] - 1) % 200 != 0 or state['epoch'] == 1:
        return
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    warmup_epoch = 11
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    lr = args.lr * (1. + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < warmup_epoch:
        lr = args.lr * max(1, current_iter - num_iter) / (warmup_iter - num_iter)
    if epoch == 0:
        lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()