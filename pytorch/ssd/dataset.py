import cv2
import io
import numpy as np
from PIL import Image
from abeja.datasets import Client

import torch
import torch.utils.data

def load_dataset_from_api(dataset_id, max_num=None, organization_id=None, credential=None):
    client = Client(organization_id, credential)
    dataset = client.get_dataset(dataset_id)
    
    if max_num is not None:
        dataset_list = dataset.dataset_items.list(prefetch=False)
        ret = []
        for d in dataset_list:
            ret.append(d)
            if len(ret) > max_num:
                break
        return ret
    else:
        return dataset.dataset_items.list(prefetch=True)


class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, index):
        if index < 0:
            raise IndexError
        for dataset in self._datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError


class DetectionDatasetFromAPI(torch.utils.data.Dataset):
    def __init__(self, dataset_list, transform=None, use_difficult=False):
        super(DetectionDatasetFromAPI, self).__init__()

        self.dataset_list = dataset_list
        self.transform = transform

        self.use_difficult = use_difficult

    def __len__(self):
        return len(self.dataset_list)


    def __getitem__(self, index):
        item = self.dataset_list[index]
        #file_content = item.source_data[0].get_content()
        file_content = None
        for i in range(5):
            try:
                file_content = item.source_data[0].get_content()
                break
            except:
                pass

        file_like_object = io.BytesIO(file_content)
       
        img = Image.open(file_like_object)
        img = np.asarray(img)
        height, width, channels = img.shape

        annotations = item.attributes['detection']

        boxes = []
        labels = []
        difficult = []
        for annotation in annotations:
            if not self.use_difficult and annotation['difficult']:
                continue

            rect = annotation['rect']
            box = ((rect['xmin'] - 1) / width, 
                  (rect['ymin'] - 1) / height, 
                  (rect['xmax'] - 1) / width,
                  (rect['ymax'] - 1) / height)

            boxes.append(box)
            labels.append(annotation['label_id'] - 1)
            difficult.append(annotation['difficult'])
        
        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.float32)
        difficult = np.array(difficult, dtype=np.bool)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
