import glob
import os
import shutil

import numpy as np

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv import utils
from chainercv.utils import read_image


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

camvid_label_colors = (
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
)
camvid_ignore_label_color = (0, 0, 0)


class CamVidDataset(GetterDataset):

    """Semantic segmentation dataset for `CamVid`_.

    .. _`CamVid`:
        https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/camvid`.
        split ({'train', 'val', 'test'}): Select from dataset splits used
            in CamVid Dataset.


    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='./SegNet-Tutorial-master/CamVid', split='train'):
        super(CamVidDataset, self).__init__()

        if split not in ['train', 'val', 'test']:
            raise ValueError(
                'Please pick split from \'train\', \'val\', \'test\'')

        img_list_path = os.path.join(data_dir, '{}.txt'.format(split))
        self.paths = [
            [os.path.join(data_dir, fn.replace('/SegNet/CamVid/', ''))
             for fn in line.split()] for line in open(img_list_path)]

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.paths)

    def _get_image(self, i):
        img_path, _ = self.paths[i]
        return img_path

    def _get_label(self, i):
        _, label_path = self.paths[i]
        return label_path
