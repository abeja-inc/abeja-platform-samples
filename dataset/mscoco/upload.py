# -*- coding: utf-8 -*-
import os
import json
import copy
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from chainercv.datasets import COCOBboxDataset
from chainercv.datasets.coco.coco_utils import get_coco
from typing import List

from abejacli.config import (
    ABEJA_PLATFORM_USER_ID,
    ABEJA_PLATFORM_TOKEN
)
from abeja.datalake import Client as DatalakeClient
from abeja.datalake.file import DatalakeFile
from abeja.datalake.channel import Channel
from abeja.datalake.storage_type import StorageType
from abeja.datasets import Client as DatasetClient
from abeja.datasets.dataset_item import DatasetItems, DatasetItem

credential = {
    'user_id': ABEJA_PLATFORM_USER_ID,
    'personal_access_token': ABEJA_PLATFORM_TOKEN
}


class COCOBobxDatasetMod(COCOBboxDataset):

    def __init__(self, data_dir='auto', split='train', year='2017',
                 use_crowded=False, return_area=False, return_crowded=False):
        super(COCOBobxDatasetMod, self).__init__(data_dir, split, year,
                                                 use_crowded, return_area, return_crowded)
        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = get_coco(split, img_split, year)
        anno_path = os.path.join(
            data_dir, 'annotations', 'instances_{}{}.json'.format(split, year))

        self.annotations = json.load(open(anno_path, 'r'))
        self.categories = copy.copy(self.annotations['categories'])

        # category id use sparse ids (.., 81, 82, 84, ..)
        self.categories.sort(key=lambda x: x['id'])

        # convert label format adapt to ABEJA Platform Datasets
        for label in self.categories:
            label['label_id'] = label['id']
            label['label'] = label.pop('name')  # if no label attribute, platform UI does not preview label name

    def _get_image(self, i):
        # filenames are not duplicated in train and val at 2017
        return Path(self.img_root) / Path(self.id_to_prop[self.ids[i]]['file_name'])


def create_dataset(client: DatasetClient, name: str, props: dict,
                   dataset_type: str, override: bool = True):
    """ creates new dataset for ABEJA platform, returns it's id """
    if override:
        response = client.datasets.list()
        datasets = {r.name: r.dataset_id for r in response}
        if name in datasets:
            client.datasets.delete(datasets[name])
    response = client.datasets.create(name, dataset_type, props=props)
    dataset_id = response.dataset_id
    return dataset_id


def build_props(categories: List[dict]) -> dict:
    props = {
        'categories': [
            {
                'category_id': 0,
                'labels': categories,
                'name': 'ms-coco'
            }
        ]
    }
    return props


def build_attributes(bboxes: list, labels: list, categories: list) -> dict:
    attributes = dict(detection=[])
    for bbox, label in zip(bboxes, labels):
        detection = {
            'category_id': 0,
            'rect': {
                'ymin': float(bbox[0]),
                'xmin': float(bbox[1]),
                'ymax': float(bbox[2]),
                'xmax': float(bbox[3]),
            },
            **categories[label]
        }
        attributes['detection'].append(detection)
    return attributes


def upload_dataset(dataset_client: DatasetClient, dataset_id: str,
                   dataset_list: list, max_workers: int = 4):
    """ uploads file info list to dataset using abeja's dataset client """
    dataset = dataset_client.get_dataset(dataset_id)

    def _f(dataset_item):
        source_data = [
            {
                'data_uri': dataset_item.source_data[0].uri,
                'data_type': dataset_item.source_data[0].type
            }
        ]  # TODO: only one url to be uploaded
        attributes = dataset_item.attributes
        dataset.dataset_items.create(source_data=source_data, attributes=attributes)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers) as executor:
            results = list(tqdm(executor.map(_f, dataset_list), total=len(dataset_list)))
        return results
    return [_f(x) for x in tqdm(dataset_list)]


def upload_image_and_attributes(channel: Channel, dataset_items: DatasetItems,
                                filepath: Path, bboxes: list, labels: list,
                                categories: List[dict]) -> DatasetItem:
    attibutes = build_attributes(bboxes, labels, categories)
    datalake_file = upload_image_to_datalake(channel, filepath)
    source_data = [
        {
            'data_uri': datalake_file.uri,
            'data_type': datalake_file.content_type
        }
    ]
    dataset_item = dataset_items.create(source_data=source_data, attributes=attibutes)
    return dataset_item


def upload_image_to_datalake(channel: Channel, filepath: Path) -> DatalakeFile:
    if filepath.suffix not in ['.jpg', '.jpeg', '.JPG', '.jpeg']:
        raise ValueError(f'invalid file format: {filepath}, it is not jpeg file')
    content_type = 'image/jpeg'
    metadata = {'filename': filepath.name}
    return channel.upload_file(str(filepath), metadata=metadata, content_type=content_type)


def upload_coco_dataset(coco_dataset: COCOBobxDatasetMod,
                        channel: Channel, dataset_items: DatasetItems, max_workers: int):
    categories = coco_dataset.categories

    def _f(x):
        filepath, bboxes, labels = x
        return upload_image_and_attributes(channel, dataset_items, filepath, bboxes, labels, categories)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers) as executor:
            results = list(tqdm(executor.map(_f, coco_dataset), total=len(coco_dataset)))
        return results
    return [_f(x) for x in tqdm(coco_dataset)]


def main(organization_id, dataset_name, max_workers):
    coco_dataset_train = COCOBobxDatasetMod(
        split='train', year='2017', use_crowded=False,
    )
    coco_dataset_val = COCOBobxDatasetMod(
        split='val', year='2017', use_crowded=False,
    )
    categories = coco_dataset_val.categories
    props = build_props(categories)

    datalake_client = DatalakeClient(organization_id=organization_id, credential=credential)
    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)

    description = f'MS-COCO detection created with dataset: {dataset_name}'
    channel = datalake_client.channels.create('', description,
                                              StorageType.DATALAKE.value)
    print(f'channel is created: {channel.channel_id}')

    print('upload train dataset...')
    dataset_id_train = create_dataset(dataset_client, dataset_name + '-train', props=props,
                                      dataset_type='detection', override=True)
    dataset_train = dataset_client.get_dataset(dataset_id_train)
    upload_coco_dataset(coco_dataset_train, channel, dataset_train.dataset_items, max_workers)

    print('upload val dataset...')
    dataset_id_val = create_dataset(dataset_client, dataset_name + '-val', props=props,
                                    dataset_type='detection', override=True)
    dataset_val = dataset_client.get_dataset(dataset_id_val)
    upload_coco_dataset(coco_dataset_val, channel, dataset_val.dataset_items, max_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload MS-COCO Dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help='dataset_name')
    parser.add_argument('--max_workers', '-w', type=int, default=4)
    args = parser.parse_args()

    organization_id = args.organization_id
    dataset_name = args.dataset_name
    max_workers = args.max_workers
    main(organization_id, dataset_name, max_workers)
