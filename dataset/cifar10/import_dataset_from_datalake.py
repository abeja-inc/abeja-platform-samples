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

logger = get_logger()


def create_request_element(channel_id, file_info, data):
    """
    create dataset item from datalake file
    :param channel_id:
    :param file_id:
    :param file_info:
    :param label_metadata_key:
    :return:
    """
    file_id = file_info.get('file_id')
    filename = data['filename']
    annotation = data['annotation']
    data_uri = 'datalake://{}/{}'.format(channel_id, file_id)

    data = {
        'source_data': [
            {
                'data_uri': data_uri,
                'data_type': file_info['content_type'],
                'metadata': {'meta-filename': filename}
            }
        ],
        'attributes': {
            'classification': annotation
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


def register_dataset_items_from_datalake(dataset_id, channel_id, label_metadata_key):
    """
    register datasets from datalake channel
    :param dataset_id: target dataset id
    :param channel_id: target channel
    :param label_metadata_key: metadata key which label value is stored
    :return:
    """
    with open('dataset.json', 'r') as f:
        dataset = json.load(f)
        label2id = {x['label']: x['label_id'] for x in dataset['attributes'][0]['categories']}
    
    def to_annotation(file_info):
        file_metadata = file_info.get('metadata')
        label = file_metadata['x-abeja-meta-{}'.format(label_metadata_key)]
        filename = file_metadata['x-abeja-meta-filename']
        return {'filename': filename, 'annotation': {label_metadata_key: label, 'label_id': label2id[label]}}

    print('Getting data from datalake....')
    file_iter = generate_channel_file_iter_by_period(channel_id)
    dataset_items = [
        create_request_element(channel_id, file_info, to_annotation(file_info))
        for file_info in file_iter
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
    parser.add_argument('--label_metadata', '-l', type=str, default='label', help='label meta data')
    args = parser.parse_args()
    
    channel_id = args.channel_id
    dataset_id = args.dataset_id
    label_metadata = args.label_metadata
    
    register_dataset_items_from_datalake(dataset_id, channel_id, label_metadata)
