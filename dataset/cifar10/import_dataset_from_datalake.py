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

credential = {
    'user_id': ABEJA_PLATFORM_USER_ID,
    'personal_access_token': ABEJA_PLATFORM_TOKEN
}

def create_request_element(channel_id, file_info, data_id, annotation):
    """
    create dataset item from datalake file
    :param channel_id:
    :param file_id:
    :param file_info:
    :param label_metadata_key:
    :return:
    """

    data_uri = 'datalake://{}/{}'.format(channel_id, file_info.file_id)
    
    data = {
        'source_data': [
            {
                'data_uri': data_uri,
                'data_type': file_info.content_type
            }
        ],
        'attributes': {
            'classification': annotation,
            'id': data_id
        }
    }
    return data


def register_dataset_items(dataset, items):
    dataset_items = dataset.dataset_items
    for item in items:
        source_data = item['source_data']
        attributes = item['attributes']
        dataset_items.create(source_data=source_data, attributes=attributes)


def register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, label_metadata_key):
    with open('dataset.json', 'r') as f:
        dataset_props = json.load(f)
    
    print('Getting data from datalake....')
    client = DatalakeClient(organization_id=organization_id, credential=credential)
    channel = client.get_channel(channel_id)
   
    def to_annotation(file_info):
        label = file_info.metadata[label_metadata_key]
        label_id = label2id[label]
        return {label_metadata_key: label, 'label_id': label_id}

    file_iter = channel.list_files(limit=1000, prefetch=False)
    label2id = {x['label']: x['label_id'] for x in dataset_props['attributes'][0]['categories']}

    dataset_items = []
    for file_info in file_iter:
        item = create_request_element(channel_id, file_info, 
                                      data_id=int(file_info.metadata['filename'].split('.')[0]),
                                      annotation=to_annotation(file_info))
        dataset_items.append(item)
        if len(dataset_items) % 1000 == 0:
            print(len(dataset_items))
    
    print('Registering dataset items....')
    dataset_params = {
        'organization_id': organization_id,
        'name': dataset_name,
        'type': 'classification',
        'props': dataset_props
    }
    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)
    dataset = dataset_client.datasets.create(dataset_name, 'classification', dataset_params)
    register_dataset_items(dataset, dataset_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload CIFAR10 dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--channel_id', '-c', type=str, required=True, help='channel_id')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help='dataset_name')
    parser.add_argument('--label_metadata', '-l', type=str, default='label', help='label meta data')
    args = parser.parse_args()
    
    organization_id = args.organization_id
    channel_id = args.channel_id
    dataset_name = args.dataset_name
    label_metadata = args.label_metadata
    
    register_dataset_items_from_datalake(organization_id, channel_id, dataset_name, label_metadata)
