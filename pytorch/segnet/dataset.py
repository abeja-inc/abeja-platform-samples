import io
from PIL import Image
import numpy as np

import torch
import torch.utils.data

from abeja.datalake import APIClient
from abeja.datalake.file import DatalakeFile
from abeja.datasets import Client


def load_dataset_from_api(dataset, max_num=None):
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


class SegmentationDatasetFromAPI(torch.utils.data.Dataset):
    def __init__(self, dataset_id, transform=None):
        super(SegmentationDatasetFromAPI, self).__init__()
        
        self.transform = transform

        client = Client()
        self.dataset = client.get_dataset(dataset_id)
        self.dataset_list = list(load_dataset_from_api(self.dataset))
       
        self.max_id = max([c['label_id']
                           for c in self.dataset.props['props']['attributes'][0]['categories']])

        self.client = APIClient()

    def __len__(self):
        return len(self.dataset_list)

    def read_data(self, source_data):
        file_content = None
        for i in range(5):
            try:
                file_content = source_data.get_content()
                break
            except:
                pass
        return file_content

    def __getitem__(self, index):
        item = self.dataset_list[index]
       
        # Image
        file_content = self.read_data(item.source_data[0])
        file_like_object = io.BytesIO(file_content)
        img = Image.open(file_like_object)
       
        # Label
        annotation = item.attributes['segmentation']
        channel_id = annotation['channel_id']
        file_id = annotation['file_id']
        uri = 'datalake://{}/{}'.format(channel_id, file_id)
        ftype = 'image/png'

        source = DatalakeFile(api=self.client, channel_id=channel_id, file_id=file_id, uri=uri, type=ftype)
        file_content = self.read_data(source)
        file_like_object = io.BytesIO(file_content)
        
        # Label id max_id is for unlabeled pixels.
        f = Image.open(file_like_object)
        try:
            limg = f.convert('P')
            label = np.asarray(limg, dtype=np.int32)
        finally:
            f.close()
        img = np.array(img)
        label[label == self.max_id] = -1

        if self.transform:
            img, label = self.transform(img, label)
        
        return torch.from_numpy(img.copy()).permute(2, 0, 1), torch.from_numpy(label.copy()).long()
