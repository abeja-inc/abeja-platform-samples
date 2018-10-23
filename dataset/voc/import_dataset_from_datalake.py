# coding: utf-8

import argparse
import json
from collections import defaultdict

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

def register_dataset_items(dataset, items):
    dataset_items = dataset.dataset_items
    for item in items:
        source_data = item['source_data']
        attributes = item['attributes']
        dataset_items.create(source_data=source_data, attributes=attributes)


def register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, split, year):
    """
    register datasets from datalake channel
    :param dataset_id: target dataset id
    :param channel_id: target channel
    :param label_metadata_key: metadata key which label value is stored
    :param max_size_for_label: max size of dataset items for each label value
    :return:
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
    for file_info in file_iter:
        if file2id(file_info) in data:
            item = create_request_element(channel_id, file_info, data[file2id(file_info)])
            dataset_items.append(item)
            if len(dataset_items) % 1000 == 0:
                print(len(dataset_items))
 
    print('Registering dataset items....')
    dataset_params = {
        'organization_id': organization_id,
        'name': dataset_name,
        'type': 'detection',
        'props': dataset_props
    }
    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)
    dataset = dataset_client.datasets.create(dataset_name, 'detection', dataset_params)
    register_dataset_items(dataset, dataset_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload VOC Dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--channel_id', '-c', type=str, required=True, help='channel_id')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help='dataset_name')
    parser.add_argument('--split', '-s', choices=['train', 'trainval', 'val', 'test'], 
                        type=str, required=True, help='split')
    parser.add_argument('--year', '-y', choices=['2007', '2012'], 
                        type=str, required=True, help='year')
    args = parser.parse_args()
    
    organization_id = args.organization_id
    channel_id = args.channel_id
    dataset_name = args.dataset_name
    split = args.split
    year = args.year
    
    register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, split, year)
