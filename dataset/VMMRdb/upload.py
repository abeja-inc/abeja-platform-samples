from typing import Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor as Executer
from tqdm import tqdm
import argparse

from abeja.datalake import Client as DatalakeClient
from abeja.datasets import Client as DatasetsClient


def build_props(image_root_dir):
    image_list = [p.name for p in image_root_dir.iterdir()]
    labels = [dict(label_id=i, label=name) for i, name in enumerate(image_list)]
    label2id = {name:i for i, name in enumerate(image_list)}
    category = {'labels': labels, 'category_id': 0, 'name': 'VMMRdb'}
    props = {"categories": [category]}
    return props, label2id


def upload_image(channel, dataset, label_id, path):
    file = channel.upload_file(str(path))
    source_data = [
        {
            'data_type': 'image/jpeg',
            'data_uri': 'datalake://{}/{}'.format(channel.channel_id, file.file_id),
        }
    ]

    data = {
        'category_id': 0,
        'label_id': label_id
    }
    attributes = {'classification': [data]}
    dataset_item = dataset.dataset_items.create(source_data=source_data, attributes=attributes)
    return dataset_item


def upload_images(channel, dataset, image_root_dir:Path, label2id:Dict[str, int], max_workers):
    
    def _upload(label, path):
        label_id = label2id[label]
        return upload_image(channel, dataset, label_id, path)
    
    executer = Executer(max_workers=max_workers)
    future_pool = []
    total = len(list(image_root_dir.iterdir()))
    print(f'total dir: {total}')
    with Executer(max_workers=max_workers) as executer:
        total = len(list(image_root_dir.iterdir()))
        future_pool = []
        for image_dir in tqdm(image_root_dir.iterdir(), total=total, position=0):
            label = image_dir.name
            def _f(p):
                return _upload(label, p)
            num_images = len(list(image_dir.iterdir()))
            future_pool  += list(tqdm(executer.map(_f, image_dir.iterdir()), total=num_images, position=1))
    return future_pool


def main(image_root_dir, channel_id, dataset_name, max_workers):
    image_root_dir = Path(image_root_dir)
    props, label2id = build_props(image_root_dir)
    
    datakale_client = DatalakeClient()
    channel = datakale_client.get_channel(channel_id)

    datasets_client = DatasetsClient()
    dataset = datasets_client.datasets.create(name=dataset_name, type='classification', props=props)
    ret = upload_images(channel, dataset, image_root_dir, label2id, max_workers=max_workers)
    print('finished!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_dir', default='VMMRdb', type=str)
    parser.add_argument('--channel_id', help='channel_id to upload images', type=str)
    parser.add_argument('--dataset_name', help='dataset name to be created', default='VMMRdb', type=str)
    parser.add_argument('--max_workers', default=1, type=int)
    args = parser.parse_args()
    main(args.image_root_dir, args.channel_id, args.dataset_name, args.max_workers)
    