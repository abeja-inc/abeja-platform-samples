from __future__ import print_function
import os
import io

import numpy as np

from PIL import Image

from abeja.datasets import Client

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from utils.callbacks import Statistics
from utils.tensorboard import Tensorboard

batch_size = 32
num_classes = 2
epochs = int(os.environ.get('NUM_EPOCHS', 100))

# input image dimensions
img_rows, img_cols = 128, 128
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
TRAINING_USE_GPU = int(os.environ.get('TRAINING_USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')

def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list

class ImageDatasetFromAPI(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_list, train=True):
        """
        the following labeling rule is used in the dataset.

         animal | label
        --------|--------
           dog  |   1
           cat  |   2
        """
        self.dataset_list = dataset_list
        self.train = train

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
        img = img.transpose((2, 0, 1))
        return img

    def preprocess_input(self, img):
        img *= (2.0 / 255.0)
        img -= 1
        return img

    def get_example(self, i):
        item = self.dataset_list[i]
        file_content = item.source_data[0].get_content()
        if isinstance(item.attributes['classification'], list):
            label = item.attributes['classification'][0]['label']
        else:
            label = item.attributes['classification']['label']
        file_like_object = io.BytesIO(file_content)
        img = self.read_image_as_array(file_like_object, target_size=(img_rows, img_cols))
        if img is None:
            return None
        img = self.preprocess_input(img)

        if self.train:
            img = img[:, :, ::-1]

        return img, int(label) - 1


class SimpleNet(chainer.Chain):
    def __init__(self):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(3, 32, 3, pad=1)
            self.b1 = L.BatchNormalization(32)
            self.c2 = L.Convolution2D(32, 32, 3, pad=1)
            self.b2 = L.BatchNormalization(32)
            self.c3 = L.Convolution2D(32, 64, 3, pad=1)
            self.b3 = L.BatchNormalization(64)
            self.c4 = L.Convolution2D(64, 128, 3, pad=1)
            self.b4 = L.BatchNormalization(128)
            self.c5 = L.Convolution2D(128, 256, 3, pad=1)
            self.b5 = L.BatchNormalization(256)
            self.l6 = L.Linear(16 * 256, 128)
            self.l7 = L.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b2(self.c2(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b3(self.c3(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b4(self.c4(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b5(self.c5(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.dropout(F.relu(self.l6(x)))

        return self.l7(x)


def handler(context):
    dataset_alias = context.datasets
    dataset_id = dataset_alias['train']   # set alias specified in console
    data = list(load_dataset_from_api(dataset_id))

    np.random.seed(0)
    data = np.random.permutation(data)
    nb_data = len(data)
    nb_train = int(7 * nb_data // 10)
    train_data_raw = data[:nb_train]
    test_data_raw = data[nb_train:]

    simple_net = SimpleNet()
    model = L.Classifier(simple_net)

    if TRAINING_USE_GPU >= 0:
        chainer.cuda.get_device(TRAINING_USE_GPU).use()  # Make a specified GPU current
        model.to_gpu()

    def make_optimizer(model, alpha=0.001, beta1=0.9):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    optimizer = make_optimizer(model)

    train_data = ImageDatasetFromAPI(train_data_raw)
    train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
    test_data = ImageDatasetFromAPI(test_data_raw)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=TRAINING_USE_GPU)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=ABEJA_TRAINING_RESULT_DIR)

    trainer.extend(extensions.Evaluator(test_iter, model, device=TRAINING_USE_GPU))

    trainer.extend(extensions.snapshot_object(simple_net, 'simple_net.model'), trigger=(epochs, 'epoch'))

    report_entries = ['epoch', 'main/loss', 'validation/main/loss',
                      'main/accuracy', 'validation/main/accuracy']

    trainer.extend(extensions.LogReport())
    trainer.extend(Statistics(report_entries, epochs), trigger=(1, 'epoch'))
    trainer.extend(Tensorboard(report_entries, out_dir=log_path))

    trainer.extend(extensions.PrintReport(report_entries))

    trainer.run()