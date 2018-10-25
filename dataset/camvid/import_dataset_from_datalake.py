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
            'segmentation': annotation
        }
    }
    return data

def register_dataset_items(dataset, items):
    dataset_items = dataset.dataset_items
    for item in items:
        source_data = item['source_data']
        attributes = item['attributes']
        dataset_items.create(source_data=source_data, attributes=attributes)

def register_dataset_items_from_datalake(organization_id, image_channel_id, label_channel_id, dataset_name, img_list_path):
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

    paths = [[fn.split('/')[-1] for fn in line.split()] for line in open(img_list_path)]
    img2label = {img: lbl for img, lbl in paths}

    print('Getting data from datalake....')
    client = DatalakeClient(organization_id=organization_id, credential=credential)
    label_channel = client.get_channel(label_channel_id)
    
    label_list = label_channel.list_files(prefetch=True)
    label2fileid = {label.metadata['filename']: label.file_id for label in label_list}
    
    image_channel = client.get_channel(image_channel_id)
    file_iter = image_channel.list_files(limit=1000, prefetch=True)
    
    dataset_items = []
    for file_info in file_iter:
        imgfile = file_info.metadata['filename']
        labelfile = img2label[imgfile]
        label_id = label2fileid[labelfile]

        annotation = {
            'channel_id': label_channel_id,
            'file_id': label_id,
            'filename': labelfile
        }
    
        item = create_request_element(image_channel_id, file_info, annotation)
        dataset_items.append(item)
 
    print('Registering dataset items....')
    dataset_params = {
        'organization_id': organization_id,
        'name': dataset_name,
        'type': 'segmentation',
        'props': dataset_props
    }
    dataset_client = DatasetClient(organization_id=organization_id, credential=credential)
    dataset = dataset_client.datasets.create(dataset_name, 'segmentation', dataset_params)
    register_dataset_items(dataset, dataset_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload CamVid Dataset')
    parser.add_argument('--organization_id', '-o', type=str, required=True, help='organization_id')
    parser.add_argument('--image_channel_id', '-i', type=str, required=True, help='channel_id of image data')
    parser.add_argument('--label_channel_id', '-l', type=str, required=True, help='channel_id of label data')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help='dataset_name')
    parser.add_argument('--img_list_path', '-p', type=str, required=True, help='annotation file')
    args = parser.parse_args()
    
    organization_id = args.organization_id
    image_channel_id = args.image_channel_id
    label_channel_id = args.label_channel_id
    dataset_name = args.dataset_name
    img_list_path = args.img_list_path
    
    register_dataset_items_from_datalake(organization_id, image_channel_id, label_channel_id, dataset_name, img_list_path)
