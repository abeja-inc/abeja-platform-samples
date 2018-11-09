import os
import io
import numpy as np
from PIL import Image

from abeja.datasets import Client

import torch
import torch.utils.data
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

from net import SimpleNet
from utils.callbacks import Statistics

batch_size = 32
num_classes = 2
epochs = int(os.environ.get('NUM_EPOCHS', 100))

# input image dimensions
img_rows, img_cols = 128, 128
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)

def load_dataset_from_api(dataset_id, max_num=None):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=False)
    
    if max_num is not None:
        ret = []
        for d in dataset_list:
            ret.append(d)
            if len(ret) > max_num:
                break
        return ret
    else:
        return dataset_list


class Preprocess(object):
    def __init__(self, train):
        self._train = train

    def __call__(self, image):
        image *= (2.0 / 255.0)
        image -= 1

        if self._train and np.random.rand() > 0.5:
            image = image[:, ::-1, :]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image.copy())


class ImageDatasetFromAPI(torch.utils.data.Dataset):
    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def read_image_as_array(self, file_obj, target_size):
        f = Image.open(file_obj)
        img = f.resize(target_size)
        try:
            img = np.asarray(img, dtype=np.float32)
        finally:
            if hasattr(f, 'close'):
                f.close()
        return img

    def __getitem__(self, index):
        item = self.dataset_list[index]
       
        file_content = None
        for i in range(5):
            try:
                file_content = item.source_data[0].get_content()
                break
            except:
                pass

        if isinstance(item.attributes['classification'], list):
            label = item.attributes['classification'][0]['label']
        else:
            label = item.attributes['classification']['label']
        file_like_object = io.BytesIO(file_content)
        
        img = self.read_image_as_array(file_like_object, target_size=(img_rows, img_cols))
        label = int(label) - 1
         
        if self.transform:
            img = self.transform(img)
        return img, label


def handler(context):
    # Data from ABEJA Platform
    dataset_alias = context.datasets
    dataset_id = dataset_alias['train']
    data = list(load_dataset_from_api(dataset_id))

    # Divide data to train adn validation
    np.random.seed(0)
    data = np.random.permutation(data)
    nb_data = len(data)
    nb_train = int(7 * nb_data // 10)
    train_data_raw = data[:nb_train]
    test_data_raw = data[nb_train:]

    # GPU
    use_cuda = USE_GPU >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = SimpleNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(net.parameters())

    # dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_data = ImageDatasetFromAPI(train_data_raw, 
                                     transform=transforms.Compose([Preprocess(train=True), ToTensor()]))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True, **kwargs)
    test_data = ImageDatasetFromAPI(test_data_raw,
                                    transform=transforms.Compose([Preprocess(train=False), ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=True, **kwargs)

    # Reporter for ABEJA Platform
    statistics = Statistics(epochs)

    for epoch in range(epochs):
        net.train()

        train_total_loss = 0
        train_total_acc = 0
        train_total = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_total_acc += (predicted == labels).sum().item()
            train_total_loss += loss.item()
        train_total_loss /= train_total
        train_total_acc /= train_total

        net.eval()

        test_total_loss = 0
        test_total_acc = 0
        test_total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_total_acc += (predicted == labels).sum().item()
                test_total_loss += loss.item()
        test_total_loss /= test_total
        test_total_acc /= test_total

        print('[{:d}] main/loss: {:.3f} main/acc: {:.3f}, main/validation/loss: {:.3f}, main/validation/acc: {:.3f}'
                .format(epoch + 1, train_total_loss, train_total_acc, test_total_loss, test_total_acc))
        
        statistics(epoch+1, train_total_loss, train_total_acc, test_total_loss, test_total_acc)
        writer.add_scalar('main/loss', train_total_loss, epoch+1)
        writer.add_scalar('main/acc', train_total_acc, epoch+1)
        writer.add_scalar('main/validation/loss', test_total_loss, epoch+1)
        writer.add_scalar('main/validation/acc', test_total_acc, epoch+1)
    
    torch.save(net.state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
