import os
import io
import copy
import numpy as np

from PIL import Image

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from abeja.datasets import Client

from utils.callbacks import Statistics
from utils.tensorboard import Tensorboard

network_model = 'ssd300'
nb_iterations = 120000

BATCHSIZE = int(os.environ.get('BATCHSIZE', '8'))
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')

def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


class DetectionDatasetFromAPI(GetterDataset):
    def __init__(self, dataset_list, use_difficult=False, return_difficult=False):
        super(DetectionDatasetFromAPI, self).__init__()

        self.dataset_list = dataset_list
        self.use_difficult = use_difficult
        
        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

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

    def _get_image(self, i):
        item = self.dataset_list[i]
        file_content = item.source_data[0].get_content()
        file_like_object = io.BytesIO(file_content)
        img = self.read_image_as_array(file_like_object)
        return img

    def _get_annotations(self, i):
        item = self.dataset_list[i]
        annotations = item.attributes['detection']

        bbox = []
        label = []
        difficult = []
        for annotation in annotations:
            if not self.use_difficult and annotation['difficult']:
                continue

            rect = annotation['rect']
            box = rect['ymin'], rect['xmin'], rect['ymax'], rect['xmax']
            bbox.append(box)
            label.append(annotation['label_id'] - 1)
            difficult.append(annotation['difficult'])
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)
        
        return bbox, label, difficult

def handler(context):
    dataset_alias = context.datasets
    trainval_2007_dataset_id = dataset_alias['trainval2007']
    trainval_2012_dataset_id = dataset_alias['trainval2012']
    test_2007_dataset_id = dataset_alias['test2007']

    trainval_2007_dataset = list(load_dataset_from_api(trainval_2007_dataset_id))
    trainval_2012_dataset = list(load_dataset_from_api(trainval_2012_dataset_id))
    test_2007_dataset = list(load_dataset_from_api(test_2007_dataset_id))

    if network_model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif network_model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if USE_GPU >= 0:
        chainer.cuda.get_device_from_id(USE_GPU).use()
        model.to_gpu()

    trainval_2007 = DetectionDatasetFromAPI(trainval_2007_dataset)
    trainval_2012 = DetectionDatasetFromAPI(trainval_2012_dataset)
    test_2007 = DetectionDatasetFromAPI(test_2007_dataset, use_difficult=True, return_difficult=True)

    train = TransformDataset(
        ConcatenatedDataset(
            trainval_2007,
            trainval_2012
        ),
        Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)

    test_iter = chainer.iterators.SerialIterator(
        test_2007, BATCHSIZE, repeat=False, shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=USE_GPU)
    trainer = training.Trainer(updater, (nb_iterations, 'iteration'), out=ABEJA_TRAINING_RESULT_DIR)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(10000, 'iteration'))

    log_interval = 100, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    print_entries = ['iteration', 
                      'main/loss', 'main/loss/loc', 'main/loss/conf',
                      'validation/main/map']
    report_entries = ['epoch', 'iteration', 'lr',
                      'main/loss', 'main/loss/loc', 'main/loss/conf',
                      'validation/main/map']
    
    trainer.extend(Statistics(report_entries, nb_iterations, obs_key='iteration'), trigger=log_interval)
    trainer.extend(Tensorboard(report_entries, out_dir=log_path))
    trainer.extend(extensions.PrintReport(print_entries), trigger=log_interval)

    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(nb_iterations, 'iteration'))

    trainer.run()
