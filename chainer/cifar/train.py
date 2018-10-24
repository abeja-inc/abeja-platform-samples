import os
import io
import numpy as np

from PIL import Image

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

import models.VGG

from abeja.datasets import Client

from utils.callbacks import Statistics
from utils.tensorboard import Tensorboard

batchsize = 64
learnrate = 0.05
epochs = 300
early_stopping = False

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')


def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list


class ImageDatasetFromAPI(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_list, train=True):
        self.dataset_list = dataset_list
        self.train = train

    def __len__(self):
        return len(self.dataset_list)

    def read_image_as_array(self, file_obj):
        img = Image.open(file_obj)
        try:
            img = np.asarray(img, dtype=np.float32)
        finally:
            if hasattr(img, 'close'):
                img.close()
        img = img.transpose((2, 0, 1))
        return img

    def get_example(self, i):
        item = self.dataset_list[i]
        file_content = item.source_data[0].get_content()
        if isinstance(item.attributes['classification'], list):
            label = item.attributes['classification'][0]['label_id']
        else:
            label = item.attributes['classification']['label_id']
        file_like_object = io.BytesIO(file_content)
        img = self.read_image_as_array(file_like_object)
        if img is None:
            return None
        img = img / 255

        if self.train:
            img = img[:, :, ::-1]

        return img, int(label)-1



def handler(context):
    class_labels = 10
    
    dataset_alias = context.datasets
    train_dataset_id = dataset_alias['train']
    test_dataset_id = dataset_alias['test']

    train_data = list(load_dataset_from_api(train_dataset_id))
    test_data = list(load_dataset_from_api(test_dataset_id))

    train = ImageDatasetFromAPI(train_data, train=True)     
    test = ImageDatasetFromAPI(test_data)
    
    net = models.VGG.VGG(class_labels)
    model = L.Classifier(net)

    if USE_GPU >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(USE_GPU).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (epochs, 'epoch')
    # Early stopping option
    if early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=early_stopping, verbose=True,
            max_trigger=(epochs, 'epoch'))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=USE_GPU)
    trainer = training.Trainer(updater, stop_trigger, out=ABEJA_TRAINING_RESULT_DIR)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=USE_GPU))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Take a snapshot at each epoch
    #trainer.extend(extensions.snapshot(
    #    filename='snaphot_epoch_{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(net, 'net.model'), trigger=(epochs, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.

    report_entries = ['epoch', 'main/loss', 'validation/main/loss',
                      'main/accuracy', 'validation/main/accuracy']

    trainer.extend(Statistics(report_entries, epochs), trigger=(1, 'epoch'))
    trainer.extend(Tensorboard(report_entries, out_dir=log_path))
    trainer.extend(extensions.PrintReport(report_entries))

    trainer.run()
