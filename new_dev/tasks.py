import os
import time
from celery import Celery
from helpers import preprocess_data, bytes_to_df
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import redis

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

try:
    r_conn = redis.Redis('redis')
    r_conn.get('test')
except redis.exceptions.ConnectionError:
    # this is a localrun
    r_conn = redis.Redis('localhost')


@celery.task(name='tasks.train')
def celery_train_model(model_id, data):
    data_df = bytes_to_df(data)

    target = data_df.iloc[:, -1]

    train_data = preprocess_data(data_df.iloc[:, :-1])

    #train_data = preprocess_data(train_data)
    model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter = 500).fit(train_data, target)

    pickled = pickle.dumps(model)

    #CONNECTION
    if r_conn.set(model_id, pickled) == True:
        return {'status': 'complete', 'unique_id': model_id}

    return {'status': 'failed', 'unique_id': model_id}
