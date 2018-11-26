import os
import catboost
from sklearn import datasets, model_selection

from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
client = Client()

def handler(context):
    iris = datasets.load_iris()
    cls = catboost.CatBoostClassifier(loss_function='MultiClass')

    X = iris.data
    y = iris.target

    data_train, data_test, label_train, label_test = model_selection.train_test_split(X, y)

    cls.fit(data_train, label_train)
    
    train_acc = cls.score(data_train, label_train)
    test_acc = cls.score(data_test, label_test)
    
    statistics = ABEJAStatistics(num_epochs=1, epoch=1)
    statistics.add_stage(ABEJAStatistics.STAGE_TRAIN, train_acc, None)
    statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION, test_acc, None)
    print(train_acc, test_acc)
    
    try:
        client.update_statistics(statistics)
    except Exception:
        pass

    cls.save_model(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'iris.mlmodel'), 
                   format='coreml', export_parameters={'prediction_type': 'probability'})
