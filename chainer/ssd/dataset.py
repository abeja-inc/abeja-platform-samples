import io
import numpy as np
from PIL import Image
from abeja.datasets import Client
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

def load_dataset_from_api(dataset_id):
    client = Client()
    dataset = client.get_dataset(dataset_id)
    dataset_list = dataset.dataset_items.list(prefetch=True)
    return dataset_list


class DetectionDatasetFromAPI(GetterDataset):
    def __init__(self, dataset_list, use_difficult=False, return_difficult=False):
        super(DetectionDatasetFromAPI, self).__init__()

        self.dataset_list = dataset_list
        self.use_difficult = use_difficult

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

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

    def _get_annotations(self, i):
        item = self.dataset_list[i]
        annotations = item.attributes['detection']

        bbox = []
        label = []
        difficult = []
        for annotation in annotations:
            if not self.use_difficult and annotation['difficult']:
                continue

            rect = annotation['rect']
            box = rect['ymin'], rect['xmin'], rect['ymax'], rect['xmax']
            bbox.append(box)
            label.append(annotation['label_id'] - 1)
            difficult.append(annotation['difficult'])
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)

        return bbox, label, difficult
