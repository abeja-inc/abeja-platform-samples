import os
import numpy as np

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

data = fetch_movielens(min_rating=5.0)
n_users, n_items = data['train'].shape

model_path = os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.npy')
model = LightFM(learning_rate=0.05, loss='warp')
model.__dict__ = np.load(model_path)[()]

def handler(_iter, ctx):
    for iter in _iter:
        user_id = iter['user_id']
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        result = {
            'user_id': user_id,
            'known_positives' : known_positives[:3],
            'recommended' : top_items[:3]
        }

        yield result
