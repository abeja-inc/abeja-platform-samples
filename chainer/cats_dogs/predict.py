import os
import chainer
import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
from preprocess import preprocess_input
from net import SimpleNet

try:
    from chainer.cuda import to_cpu
except ImportError:
    pass

num_classes = 2
img_rows, img_cols = 128, 128
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

model = SimpleNet(num_classes)
chainer.serializers.load_npz(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'simple_net.model'), model)

if USE_GPU >= 0:
    chainer.cuda.get_device(USE_GPU).use()  # Make a specified GPU current
    model.to_gpu()

def decode_predictions(result):
    categories = {
        0: 'dog',
        1: 'cat'
    }
    result_with_labels = [{"label": categories[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)

def handler(_iter, ctx):
    for img in _iter:
        img = Image.fromarray(img)
        img = img.resize((img_rows, img_cols))

        x = np.asarray(img)
        x = np.asarray(x, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                predict = F.softmax(model(x))
                result = predict[0].data
                if USE_GPU >= 0:
                    result = to_cpu(result)

        sorted_result = decode_predictions(result.tolist())
        yield {"result": sorted_result}
