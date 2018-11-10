import io
from PIL import Image
import numpy as np
from abeja.datasets import Client

import torch
import torch.utils.data


def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list


class ImageDatasetFromAPI(torch.utils.data.Dataset):
    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        item = self.dataset_list[index]

        file_content = None
        for i in range(5):
            try:
                file_content = item.source_data[0].get_content()
                break
            except:
                pass

        if isinstance(item.attributes['classification'], list):
            label = item.attributes['classification'][0]['label_id']
        else:
            label = item.attributes['classification']['label_id']
        file_like_object = io.BytesIO(file_content)
        
        img = Image.open(file_like_object)
        label = int(label) - 1
        
        if self.transform:
            img = self.transform(img)

        return img, label
