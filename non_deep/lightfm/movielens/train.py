import os
import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
client = Client()

def handler(context):
    data = fetch_movielens(min_rating=5.0)

    model = LightFM(loss='warp')

    epochs = 50
    
    for epoch in range(1, epochs + 1):
        print('Epoch: {}'.format(epoch))
        model.fit_partial(data['train'], epochs=1, num_threads=1)
        
        train_acc = precision_at_k(model, data['train'], k=5).mean()
        test_acc = precision_at_k(model, data['test'], k=5).mean()
        print("Train precision: {}".format(train_acc))
        print("Test precision: {}".format(test_acc))
        
        statistics = ABEJAStatistics(num_epochs=epochs, epoch=epoch)
        statistics.add_stage(ABEJAStatistics.STAGE_TRAIN, float(train_acc), None)
        statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION, float(test_acc), None)
        
        try:
            client.update_statistics(statistics)
        except Exception:
            pass

    np.save(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.npy'), model.__dict__)
