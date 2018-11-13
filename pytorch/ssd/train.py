import os

import torch
import torch.optim as optim
import torch.utils.data as data

import tools
from config import MEANS, PARAMS

from ssd import build_ssd
from augmentations import SSDAugmentation
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss

from dataset import load_dataset_from_api
from dataset import ConcatenatedDataset
from dataset import DetectionDatasetFromAPI

from tensorboardX import SummaryWriter
from utils.callbacks import Statistics

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)

use_cuda = USE_GPU >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 32
lr = 1e-3
num_classes = 21
lr_steps = (80000, 100000, 120000)
max_iter = 120000
min_dim = 300

statistics = Statistics(max_iter)
    
def download(url, filename):
    import urllib
    urllib.request.urlretrieve(url, filename)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval(testloader, ssd_net, criterion, iteration):
    ssd_net.eval()
    
    test_total = 0
    test_l = 0
    test_c = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            # load train data
            images = images.to(device)
            targets = [ann.to(device) for ann in targets] 
            
            # forward
            out = ssd_net(images)
            
            # backprop
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            
            test_total += out[0].size(0)
            test_loss += loss.item()
            test_l += loss_c.item()
            test_c += loss_l.item()

        test_loss /= test_total
        test_l /= test_total
        test_c /= test_total

        print('[Test] iter {}, loss: {:.4f}'.format(iteration, test_loss))
        statistics(iteration, None, None, test_loss, None)
        writer.add_scalar('test/loss', test_loss, iteration)
        writer.add_scalar('test/loc_loss', test_l, iteration)
        writer.add_scalar('test/conf_loss', test_c, iteration)


def handler(context):
    dataset_alias = context.datasets
    
    trainval_2007_dataset_id = dataset_alias['trainval2007']
    trainval_2012_dataset_id = dataset_alias['trainval2012']
    test_2007_dataset_id = dataset_alias['trainval2007']

    trainval_2007_dataset = list(load_dataset_from_api(trainval_2007_dataset_id))
    trainval_2012_dataset = list(load_dataset_from_api(trainval_2012_dataset_id))
    test_2007_dataset = list(load_dataset_from_api(test_2007_dataset_id))
    
    trainval_2007 = DetectionDatasetFromAPI(trainval_2007_dataset, transform=SSDAugmentation(min_dim, MEANS))
    trainval_2012 = DetectionDatasetFromAPI(trainval_2012_dataset, transform=SSDAugmentation(min_dim, MEANS))
    test_2007 = DetectionDatasetFromAPI(trainval_2007_dataset, transform=SSDAugmentation(min_dim, MEANS))
    train_dataset = ConcatenatedDataset(trainval_2007, trainval_2012)
    test_dataset = test_2007
    
    priorbox = PriorBox(min_dim, PARAMS)
    with torch.no_grad():
        priors = priorbox.forward().to(device)

    ssd_net = build_ssd('train', priors, min_dim, num_classes)
    ssd_net = ssd_net.to(device)

    url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
    weight_file = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'vgg16_reducedfc.pth')
    download(url, weight_file)
    
    vgg_weights = torch.load(weight_file)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

    optimizer = optim.SGD(ssd_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, PARAMS['variance'], device)

    # loss counters
    step_index = 0

    trainloader = data.DataLoader(train_dataset, batch_size,
                                  num_workers=0,
                                  shuffle=True, collate_fn=tools.detection_collate,
                                  pin_memory=True)

    testloader = data.DataLoader(test_dataset, batch_size,
                                 num_workers=0,
                                 shuffle=False, collate_fn=tools.detection_collate,
                                 pin_memory=True)

    # create batch iterator
    iteration = 1
    while iteration <= max_iter:
        ssd_net.train()
        for images, targets in trainloader:
            if iteration > max_iter:
                break

            if iteration in lr_steps:
                step_index += 1
                adjust_learning_rate(optimizer, 0.1, step_index)

            # load train data
            images = images.to(device)
            targets = [ann.to(device) for ann in targets] 
            
            # forward
            out = ssd_net(images)
            
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print('[Train] iter {}, loss: {:.4f}'.format(iteration, loss.item()))
                statistics(iteration, loss.item(), None, None, None)
                writer.add_scalar('main/loss', loss.item(), iteration)
                writer.add_scalar('main/loc_loss', loss_l.item(), iteration)
                writer.add_scalar('main/conf_loss', loss_c.item(), iteration)

            if iteration % 10000 == 0:
                eval(testloader, ssd_net, criterion, iteration)
                ssd_net.train()

            iteration += 1
    torch.save(ssd_net.state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
