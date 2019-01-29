# coding: utf-8
import argparse
import json
from collections import defaultdict
from typing import List
from pathlib import PurePath
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from abejacli.config import (
    ABEJA_PLATFORM_USER_ID,
    ABEJA_PLATFORM_TOKEN
)
from abeja.datalake import Client as DatalakeClient
from abeja.datalake.channel import Channel
from abeja.datalake.storage_type import StorageType
from abeja.datasets import Client as DatasetClient
from abeja.datasets.dataset_item import DatasetItems

from pet import load_data, PetData, int_to_category

credential = {
    'user_id': ABEJA_PLATFORM_USER_ID,
    'personal_access_token': ABEJA_PLATFORM_TOKEN
}


def create_request_element(channel_id, file_info, annotation):
    """
    create dataset item from datalake file

    Args:
        channel_id:
        file_info:
        annotation:

    Returns:
        data(dict):

    """
    file_id = file_info.file_id
    data_uri = 'datalake://{}/{}'.format(channel_id, file_id)

    attributes = {
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
    return attributes


def register_dataset_items(dataset, items):
    dataset_items = dataset.dataset_items
    for item in items:
        source_data = item['source_data']
        attributes = item['attributes']
        dataset_items.create(source_data=source_data, attributes=attributes)


def create_dataset(dataset_client, name, props):
    res = dataset_client.datasets.create(name, type='detection', props=props)
    dataset_id = res.dataset_id
    return dataset_id


def build_attributes(datum: PetData):
    attributes = {
        "detection": [{
            'category_id': 0,
            'label_id': datum.class_id,
            'label': int_to_category[datum.class_id],
            'rect': datum.bbox,
            'species': datum.species,
            'size': datum.size
        }]
    }

    return attributes


def upload_datum(channel: Channel, dataset_items: DatasetItems, datum: PetData, is_train: bool):
    # upload file to the channel
    metadata = {
        'filename': PurePath(datum.image_path).name,
        'type': 'trainval' if is_train else 'test'
    }

    response = channel.upload_file(datum.image_path, metadata=metadata)

    # upload item to the datase
    attributes = build_attributes(datum)
    source_data = [{
        "data_type": response.content_type,
        "data_uri": response.uri,
    }]

    res = dataset_items.create(source_data=source_data, attributes=attributes)
    return res


def upload_data(channel: Channel, dataset_items: DatasetItems, pet_dataset: List[PetData], is_train: bool,
                max_workers: int = 1):
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers) as executor:
            _f = lambda x: upload_datum(channel, dataset_items, x, is_train)
            results = list(tqdm(executor.map(_f, pet_dataset), total=len(pet_dataset)))
        return results

    results = [upload_datum(channel, dataset_items, datum, is_train) for datum in tqdm(pet_dataset)]
    return results


def register_dataset(organization_id, datalake_name, dataset_name, dataset_json_path, max_workers):
    """
    register datasets from datalake channel
    """
    print('prepare PET dataset')
    pet_dataset_trainval, pet_dataset_test = load_data()
    print(f'num of trainval: {len(pet_dataset_trainval)} test: {len(pet_dataset_trainval)}')

    print('create channel')
    description = 'The Oxford-IIIT Pet Dataset'
    datalake_client = DatalakeClient(organization_id=organization_id, credential=credential)
    channel = datalake_client.channels.create(datalake_name, description, StorageType.DATALAKE.value)
    print(f'channel created: {channel.channel_id}')

    print('register datasets')
    with open(dataset_json_path, 'r') as f:
        dataset_format = json.load(f)

    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)

    dataset_trainval = dataset_client.datasets.create(
        dataset_name + '-trainval', dataset_format['type'], dataset_format['props'])
    print(f'trainval dataset created: {dataset_trainval.dataset_id}')

    dataset_test = dataset_client.datasets.create(
        dataset_name + '-test', dataset_format['type'], dataset_format['props'])
    print(f'test dataset created: {dataset_test.dataset_id}')

    print('start uploading trainval..')
    upload_data(channel, dataset_trainval.dataset_items, pet_dataset_trainval, is_train=True, max_workers=max_workers)

    print('start uploading test..')
    upload_data(channel, dataset_test.dataset_items, pet_dataset_test, is_train=False, max_workers=max_workers)

    print('finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload VOC Dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--datalake_name', type=str, required=True, help='datalake_name')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--format_file', type=str, default='dataset.json', help='path of dataset.json')
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    organization_id = args.organization_id
    datalake_name = args.datalake_name
    dataset_name = args.dataset_name
    dataset_json_path = args.format_file
    max_workers = args.max_workers

    register_dataset(organization_id, datalake_name, dataset_name, dataset_json_path, max_workers)
