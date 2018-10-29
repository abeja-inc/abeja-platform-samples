import io
from PIL import Image
import chainer
import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

from abeja.datalake import APIClient
from abeja.datalake.file import DatalakeFile
from abeja.datasets import Client

class SegmentationDatasetFromAPI(GetterDataset):
    def __init__(self, dataset_id, use_difficult=False, return_difficult=False):
        super(SegmentationDatasetFromAPI, self).__init__()

        client = Client()
        self.dataset = client.get_dataset(dataset_id)

        self.dataset_list = list(self.dataset.dataset_items.list(prefetch=True))
        
        self.max_id = max([c['label_id']
                           for c in self.dataset.props['props']['attributes'][0]['categories']])

        self.client = APIClient()
        self.add_getter('img', self._get_image)
        self.add_getter('iabel', self._get_label)

    def __len__(self):
        return len(self.dataset_list)

    def read_image_as_array(self, file_obj):
        img = Image.open(file_obj)
        try:
            img = np.asarray(img, dtype=np.float32)
        finally:
            if hasattr(img, 'close'):
                img.close()
        img = img.transpose((2, 0, 1))
        return img

    def _get_image(self, i):
        item = self.dataset_list[i]
        file_content = item.source_data[0].get_content()
        file_like_object = io.BytesIO(file_content)
        img = self.read_image_as_array(file_like_object)
        return img

    def _get_label(self, i):
        item = self.dataset_list[i]
        annotation = item.attributes['segmentation']
        channel_id = annotation['channel_id']
        file_id = annotation['file_id']
        uri = 'datalake://{}/{}'.format(channel_id, file_id)
        ftype = 'image/png'

        source = DatalakeFile(api=self.client, channel_id=channel_id, file_id=file_id, uri=uri, type=ftype)
        file_content = source.get_content()
        file_like_object = io.BytesIO(file_content)

        f = Image.open(file_like_object)
        try:
            img = f.convert('P')
            label = np.asarray(img, dtype=np.int32)
        finally:
            f.close()

        # Label id max_id is for unlabeled pixels.
        label[label == self.max_id] = -1

        return label
