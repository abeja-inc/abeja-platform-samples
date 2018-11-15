"""Training code of SegNet
Original author: Preferred Networks, Inc.
https://github.com/chainer/chainercv/blob/master/examples/ssd/train.p://github.com/chainer/chainercv/blob/master/examples/segnet/train.py
Updated by: ABEJA, Inc.
"""

import os
import io
import numpy as np

from PIL import Image

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

from augmentations import SegNetAugmentation
from segnet import SegNet
from segnet import PixelwiseSoftmaxClassifier

from abeja.datalake import APIClient
from abeja.datasets import Client
from abeja.datalake.file import DatalakeFile
from utils.callbacks import Statistics
from dataset import SegmentationDatasetFromAPI
from tensorboardX import SummaryWriter


BATCHSIZE = int(os.environ.get('BATCHSIZE', '4'))
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))
lr = float(os.environ.get('LEARNING_RATE', 0.1))
MEANS = (104, 117, 123)

use_cuda = USE_GPU >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

epochs = int(os.environ.get('NUM_EPOCHS', 450))
log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)

camvid_label_names = (
    'Sky',
    'Building',
    'Pole',
    'Road',
    'Pavement',
    'Tree',
    'SignSymbol',
    'Fence',
    'Car',
    'Pedestrian',
    'Bicyclist',
)

def calc_weight(dataset):
    n_class = dataset.max_id
    n_cls_pixels = np.zeros((n_class,))
    n_img_pixels = np.zeros((n_class,))
    
    for img, label in dataset:
        label = label.numpy()
        for cls_i in np.unique(label):
            if cls_i == -1:
                continue
            n_cls_pixels[cls_i] += np.sum(label == cls_i)
            n_img_pixels[cls_i] += label.size
    freq = (n_cls_pixels + 1) / (n_img_pixels + 1)
    median_freq = np.median(freq)
    class_weight = median_freq / freq
    
    return torch.from_numpy(class_weight).float()

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
        train_acc += predicted.eq(targets).sum().item() / predicted.numpy().size
        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(batch_idx)
            print('[{:d} Iter] main/loss: {:.3f} main/acc: {:.3f}'
                .format(batch_idx, train_loss/train_total, train_acc/train_total))

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
            test_acc += predicted.eq(targets).sum().item() / predicted.numpy().size
            
            if batch_idx % 10 == 0:
                print(batch_idx)
                print('[{:d} Iter] main/loss: {:.3f} main/acc: {:.3f}'
                    .format(batch_idx, test_loss/test_total, test_acc/test_total))

    test_loss /= test_total
    test_acc /= test_total

    return test_loss, test_acc

def handler(context):
    # Dataset
    dataset_alias = context.datasets
    train_dataset_id = dataset_alias['train']
    val_dataset_id = dataset_alias['val']
    
    trainset = SegmentationDatasetFromAPI(train_dataset_id, transform=SegNetAugmentation(MEANS))
    valset = SegmentationDatasetFromAPI(val_dataset_id, transform=SegNetAugmentation(MEANS, False))
    class_weight = calc_weight(SegmentationDatasetFromAPI(train_dataset_id, transform=SegNetAugmentation(MEANS, False)))
    class_weight = class_weight.to(device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCHSIZE, shuffle=False, num_workers=0)

    # Model
    net = SegNet(3, n_class=len(camvid_label_names))
    net = net.to(device)
    
    # Optimizer
    #criterion = PixelwiseSoftmaxClassifier(weight=class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, ignore_index=-1)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    statistics = Statistics(epochs)
    
    for epoch in range(epochs):
        scheduler.step()

        train_loss, train_acc = train(net, optimizer, trainloader, criterion, epoch)
        test_loss, test_acc = test(net, valloader, criterion, epoch)
        
        # Reporting
        print('[{:d}] main/loss: {:.3f} main/acc: {:.3f}, main/validation/loss: {:.3f}, main/validation/acc: {:.3f}'
                .format(epoch + 1, train_loss, train_acc, test_loss, test_acc))
 
        statistics(epoch+1, train_loss, train_acc, test_loss, test_acc)
        writer.add_scalar('main/loss', train_loss, epoch+1)
        writer.add_scalar('main/acc', train_acc, epoch+1)
        writer.add_scalar('main/validation/loss', test_loss, epoch+1)
        writer.add_scalar('main/validation/acc', test_acc, epoch+1)

    torch.save(net.state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
