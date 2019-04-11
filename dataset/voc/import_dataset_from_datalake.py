# coding: utf-8

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from abejacli.config import (
    ABEJA_PLATFORM_USER_ID,
    ABEJA_PLATFORM_TOKEN
)
from abeja.datalake import Client as DatalakeClient
from abeja.datasets import Client as DatasetClient
import voc_bbox_dataset

credential = {
    'user_id': ABEJA_PLATFORM_USER_ID,
    'personal_access_token': ABEJA_PLATFORM_TOKEN
}


def create_request_element(channel_id, file_info, annotation):
    """
    create dataset item from datalake file
    :param channel_id:
    :param file_id:
    :param file_info:
    :param label_metadata_key:
    :return:
    """
    file_id = file_info.file_id
    data_uri = 'datalake://{}/{}'.format(channel_id, file_id)

    data = {
        'source_data': [
            {
                'data_uri': data_uri,
                'data_type': file_info.content_type
            }
        ],
        'attributes': {
            'detection': annotation
        }
    }
    return data


def register_dataset_items(dataset, items, max_workers: int):
    dataset_items = dataset.dataset_items

    def _f(x):
        source_data = x['source_data']
        attributes = x['attributes']
        return dataset_items.create(source_data=source_data, attributes=attributes)

    if max_workers > 1:
        print(f'using workers: {max_workers}')
        with ThreadPoolExecutor(max_workers) as executor:
            results = list(tqdm(executor.map(_f, items), total=len(items)))
        return results

    return [_f(item) for item in tqdm(items)]


def register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, split, year, max_workers):
    """
    register datasets from datalake channel

    Args:
        organization_id:
        channel_id:
        dataset_name:
        split:
        year:
        max_workers:

    Returns:

    """

    with open('dataset.json', 'r') as f:
        dataset_props = json.load(f)

    voc_dataset = voc_bbox_dataset.VOCBboxDataset(split=split, year=year)
    nb_data = len(voc_dataset)

    data = {}
    for i in range(nb_data):
        id, annotation = voc_dataset.get_annotations(i)
        data[id] = annotation

    print('Getting data from datalake....')
    client = DatalakeClient(organization_id=organization_id, credential=credential)
    channel = client.get_channel(channel_id)

    def file2id(file_info):
        return file_info.metadata['filename'].split('.')[0]

    file_iter = channel.list_files(limit=1000, prefetch=False)
    dataset_items = []
    for file_info in tqdm(file_iter):
        if file2id(file_info) in data:
            item = create_request_element(channel_id, file_info, data[file2id(file_info)])
            dataset_items.append(item)

    print('Registering dataset items....')
    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)
    dataset = dataset_client.datasets.create(dataset_name, dataset_props['type'], dataset_props['props'])
    register_dataset_items(dataset, dataset_items, max_workers=max_workers)
    print('uploaded!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload VOC Dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--channel_id', '-c', type=str, required=True, help='channel_id')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help='dataset_name')
    parser.add_argument('--split', '-s', choices=['train', 'trainval', 'val', 'test'],
                        type=str, required=True, help='split')
    parser.add_argument('--year', '-y', choices=['2007', '2012'],
                        type=str, required=True, help='year')
    parser.add_argument('--max_workers', type=int, default=1, help='max_workers')
    args = parser.parse_args()

    organization_id = args.organization_id
    channel_id = args.channel_id
    dataset_name = args.dataset_name
    split = args.split
    year = args.year
    max_workers = args.max_workers

    register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, split, year, max_workers)
