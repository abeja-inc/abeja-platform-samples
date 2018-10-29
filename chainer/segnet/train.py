import os
import io
import numpy as np

from PIL import Image

import chainer
import numpy as np

from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from chainercv.datasets import camvid_label_names
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import SegNetBasic
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

from abeja.datalake import APIClient
from abeja.datasets import Client
from abeja.datalake.file import DatalakeFile
from utils.callbacks import Statistics
from utils.tensorboard import Tensorboard
from dataset import SegmentationDatasetFromAPI

nb_iterations = 16000

BATCHSIZE = int(os.environ.get('BATCHSIZE', '12'))
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')

def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def calc_weight(dataset):
    n_class = dataset.max_id
    n_cls_pixels = np.zeros((n_class,))
    n_img_pixels = np.zeros((n_class,))
    
    for img, label in dataset:
        for cls_i in np.unique(label):
            if cls_i == -1:
                continue
            n_cls_pixels[cls_i] += np.sum(label == cls_i)
            n_img_pixels[cls_i] += label.size
    freq = n_cls_pixels / n_img_pixels
    median_freq = np.median(freq)
    class_weight = median_freq / freq
    
    return class_weight

def handler(context):
    # Triggers
    log_trigger = (50, 'iteration')
    validation_trigger = (2000, 'iteration')
    end_trigger = (nb_iterations, 'iteration')

    # Dataset
    dataset_alias = context.datasets
    train_dataset_id = dataset_alias['train']
    val_dataset_id = dataset_alias['val']
    train = SegmentationDatasetFromAPI(train_dataset_id)
    val = SegmentationDatasetFromAPI(val_dataset_id)
    class_weight = calc_weight(train)

    print(class_weight)
    
    train = TransformDataset(train, transform)

    # Iterator
    train_iter = iterators.SerialIterator(train, BATCHSIZE)
    val_iter = iterators.SerialIterator(
        val, BATCHSIZE, shuffle=False, repeat=False)

    # Model
    model = SegNetBasic(n_class=len(camvid_label_names))
    model = PixelwiseSoftmaxClassifier(
        model, class_weight=class_weight)
    
    if USE_GPU >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(USE_GPU).use()
        model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    # Updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=USE_GPU)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=ABEJA_TRAINING_RESULT_DIR)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_iteration-{.updater.iteration}'),
        trigger=end_trigger)
    
    print_entries = [
        'iteration', 'main/loss', 'validation/main/miou',
        'validation/main/mean_class_accuracy',
        'validation/main/pixel_accuracy']
   
    report_entries = [
        'epoch', 'iteration', 'lr',
        'main/loss', 'validation/main/miou',
        'validation/main/mean_class_accuracy',
        'validation/main/pixel_accuracy']

    trainer.extend(Statistics(report_entries, nb_iterations, obs_key='iteration'), trigger=log_trigger)
    trainer.extend(Tensorboard(report_entries, out_dir=log_path))
    trainer.extend(extensions.PrintReport(print_entries), trigger=log_trigger)

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model.predictor,
            camvid_label_names),
        trigger=validation_trigger)

    trainer.run()
