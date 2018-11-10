"""Training code for CIFAR10 dataset
Original author: kuangliu
https://github.com/kuangliu/pytorch-cifar
Updated by: ABEJA, Inc.
"""

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

from models import *
from utils.callbacks import Statistics
from dataset import load_dataset_from_api
from dataset import ImageDatasetFromAPI

lr = float(os.environ.get('LEARNING_RATE', 0.1))
epochs = int(os.environ.get('NUM_EPOCHS', 450))
model = str(os.environ.get('MODEL', 'GoogLeNet'))

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)

use_cuda = USE_GPU >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Training
def train(net, optimizer, trainloader, criterion, epoch):
    net.train()
    train_loss = 0
    train_acc = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_acc += predicted.eq(targets).sum().item()
        train_loss += loss.item()

    train_loss /= train_total
    train_acc /= train_total

    return train_loss, train_acc

def test(net, testloader, criterion, epoch):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_acc += predicted.eq(targets).sum().item()
    test_loss /= test_total
    test_acc /= test_total

    return test_loss, test_acc

def handler(context):
    dataset_alias = context.datasets
    train_dataset_id = dataset_alias['train']
    test_dataset_id = dataset_alias['test']

    train_data = list(load_dataset_from_api(train_dataset_id))
    test_data = list(load_dataset_from_api(test_dataset_id))
 
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset = ImageDatasetFromAPI(train_data, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testset = ImageDatasetFromAPI(test_data, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    print('==> Building model..')
    if model == 'VGG19':
        net = VGG('VGG19')
    elif model == 'ResNet18':
        net = ResNet18()
    elif model == 'PreActResNet18':
        net = PreActResNet18()
    elif model == 'GoogLeNet':
        net = GoogLeNet()
    elif model == 'DenseNet121':
        net = DenseNet121()
    elif model == 'ResNeXt29_2x64d':
        net = ResNeXt29_2x64d()
    elif model == 'MobileNet':
        net = MobileNet()
    elif model == 'MobileNetV2':
        net = MobileNetV2()
    elif model == 'DPN92':
        net = DPN92()
    elif model == 'ShuffleNetG2':
        net = ShuffleNetG2()
    elif model == 'SENet18':
        net = SENet18()
    elif model == 'ShuffleNetV2':
        net = ShuffleNetV2(1)
    
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    statistics = Statistics(epochs)
    
    for epoch in range(epochs):
        scheduler.step()

        train_loss, train_acc = train(net, optimizer, trainloader, criterion, epoch)
        test_loss, test_acc = test(net, testloader, criterion, epoch)
        
        # Reporting
        print('[{:d}] main/loss: {:.3f} main/acc: {:.3f}, main/validation/loss: {:.3f}, main/validation/acc: {:.3f}'
                .format(epoch + 1, train_loss, train_acc, test_loss, test_acc))
 
        statistics(epoch+1, train_loss, train_acc, test_loss, test_acc)
        writer.add_scalar('main/loss', train_loss, epoch+1)
        writer.add_scalar('main/acc', train_acc, epoch+1)
        writer.add_scalar('main/validation/loss', test_loss, epoch+1)
        writer.add_scalar('main/validation/acc', test_acc, epoch+1)

    torch.save(net.state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
