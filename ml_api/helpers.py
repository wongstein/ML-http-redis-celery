from datetime import datetime
import pickle
from sklearn import preprocessing
from io import StringIO
import pandas as pd

def gen_model_unique_id():
    #use datetime to generate unique id
    #ASSUMING: This isn't a distributed system, aka it's unlikely that models are being trained at the exact same time.
    return 'model_' + datetime.now().strftime('%Y%m%d-%H%M%S')

def gen_model_task_unique_id(model_id):
    return model_id + "_task"

def preprocess_data(data_x):
    return preprocessing.scale(data_x)

def bytes_to_df(data_b):
    if type(data_b) == type(b'byte'):
        data_b = data_b.decode('utf-8')
    data_s = StringIO(data_b)
    data_df = pd.read_csv(data_s, header = None)

    return data_df
