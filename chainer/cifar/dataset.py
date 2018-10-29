import io
from PIL import Image
import numpy as np
import chainer
from abeja.datasets import Client


def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list


class ImageDatasetFromAPI(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_list, train=True):
        self.dataset_list = dataset_list
        self.train = train

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

    def get_example(self, i):
        item = self.dataset_list[i]
        file_content = item.source_data[0].get_content()
        if isinstance(item.attributes['classification'], list):
            label = item.attributes['classification'][0]['label_id']
        else:
            label = item.attributes['classification']['label_id']
        file_like_object = io.BytesIO(file_content)
        img = self.read_image_as_array(file_like_object)
        if img is None:
            return None
        img = img / 255

        if self.train:
            img = img[:, :, ::-1]

        return img, int(label)-1
