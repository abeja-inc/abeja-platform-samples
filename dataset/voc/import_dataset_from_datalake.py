# coding: utf-8

import argparse
import json
from collections import defaultdict
from abejacli.config import (
    ORGANIZATION_ENDPOINT
)
from abejacli.session import (
    api_post
)
from abejacli.logger import get_logger
from abejacli.datalake import generate_channel_file_iter_by_period
from abejacli.config import DATASET_CHUNK_SIZE

import voc_bbox_dataset

logger = get_logger()

def create_request_element(channel_id, file_info, annotation):
    """
    create dataset item from datalake file
    :param channel_id:
    :param file_id:
    :param file_info:
    :param label_metadata_key:
    :return:
    """
    file_id = file_info.get('file_id')
    data_uri = 'datalake://{}/{}'.format(channel_id, file_id)

    data = {
        'source_data': [
            {
                'data_uri': data_uri,
                'data_type': file_info['content_type']
            }
        ],
        'attributes': {
            'detection': annotation
        }
    }
    return data

def register_dataset_items(dataset_id, items):
    """
    execute dataset api to registr dataset items
    :param dataset_id:
    :param items:
    :return:
    """
    url = '{}/datasets/{}/items'.format(ORGANIZATION_ENDPOINT, dataset_id)

    def _chunked(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # max number of items for add items should is 500 (by default)
    for chunked_items in _chunked(items, DATASET_CHUNK_SIZE):
        api_post(url, json.dumps(chunked_items))


def register_dataset_items_from_datalake(dataset_id, channel_id, split, year):
    """
    register datasets from datalake channel
    :param dataset_id: target dataset id
    :param channel_id: target channel
    :param label_metadata_key: metadata key which label value is stored
    :param max_size_for_label: max size of dataset items for each label value
    :return:
    """
    dataset = voc_bbox_dataset.VOCBboxDataset(split=split, year=year)
    nb_data = len(dataset)
    
    data = {}
    for i in range(nb_data):
        id, annotation = dataset.get_annotations(i)
        data[id] = annotation

    to_id = lambda x: x['metadata']['x-abeja-meta-filename'].split('.')[0]
        
    print('Getting data from datalake....')
    file_iter = generate_channel_file_iter_by_period(channel_id)
    dataset_items = [
        create_request_element(channel_id, file_info, data[to_id(file_info)])
        for file_info in file_iter if to_id(file_info) in data
    ]
    print('Registering dataset items....')
    register_dataset_items(dataset_id, dataset_items)
    return {
        'result': 'success',
        'dataset_items': len(dataset_items),
        'dataset_id': dataset_id,
        'channel_id': channel_id
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload VOC Dataset')
    parser.add_argument('--channel_id', '-c', type=str, required=True, help='channel_id')
    parser.add_argument('--dataset_id', '-d', type=str, required=True, help='dataset_id')
    parser.add_argument('--split', '-s', choices=['train', 'trainval', 'val', 'test'], 
                        type=str, required=True, help='split')
    parser.add_argument('--year', '-y', choices=['2007', '2012'], 
                        type=str, required=True, help='year')
    args = parser.parse_args()
    
    channel_id = args.channel_id
    dataset_id = args.dataset_id
    split = args.split
    year = args.year
    
    register_dataset_items_from_datalake(dataset_id, channel_id, split, year)
