import flask
from flask import Flask, url_for
from celery import Celery
import celery.states as states

import json

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from helpers import gen_model_unique_id, gen_model_task_unique_id, preprocess_data
import redis

app = Flask(__name__)

#celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

#initialize Celery
celery = Celery(app.name, broker = app.config['CELERY_BROKER_URL'],
                backend = app.config['CELERY_RESULT_BACKEND'])

#Redis connection
r_conn = redis.Redis('localhost')

####################### CELERY TASKS ##############################################

#Expecting data to come in byte-string
@celery.task(name='tasks.train')
def celery_train_model(model_id, data):
    data_df = bytes_to_df(data)

    target = data_df.iloc[:, -1]

    train_data = preprocess_data(data_df.iloc[:, :-1])

    #train_data = preprocess_data(train_data)
    model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter = 500).fit(train_data, target)

    pickled = pickle.dumps(model)

    #CONNECTION
    conn = redis.Redis('localhost')
    if conn.set(model_id, pickled) == True:
        return {'status': 'complete', 'unique_id': model_id}

    return {'status': 'failed', 'unique_id': model_id}

########### END POINTS #########################################################
@app.route('/check_model_status/<string:model_id>')
def check_model_status(model_id):
    model_task_id = gen_model_task_unique_id(model_id)
    task_id = r_conn.get(model_task_id)

    if task_id: #if training or if training is completed
        return check_task(task_id)

    response_message = "The model id you supplied either doesn't exist or the model_id provided is wrong."

    return flask.Response(response = response_message,
                        status = 404,
                        mimetype = 'text/plain')


@app.route('/models', methods = ['POST'])
def train_model_endpoint():
    data = get_request_data()

    if type(data) == type('string'):
        return flask.Response(response = data,
                              status = 415,
                              mimetype = 'text/plain')

    model_id = gen_model_unique_id()
    task = celery.send_task('tasks.train', args = [model_id, data.decode('utf-8')], kwargs={})

    #save task id to model id
    model_task_key = gen_model_task_unique_id(model_id)
    r_conn.set(model_task_key, task.id)

    response_message = "You're model is being trained right now, you can check it's status by hitting the check_model_status endpoint"
    response = { 'message': response_message,
                 'model_id': model_id,
                 'task_id': task.id}
    return flask.Response(response = json.dumps(response),
                          status = 200,
                          mimetype = 'text/plain')

#expecting
#added post to route for easy testing in PostMan
@app.route('/models/<string:model_id>', methods = ['GET'])
def predict(model_id):
    #find model
    model_s = r_conn.get(model_id)

    # when model doesn't exist
    if model_s == None:
        return check_model_status(model_id)

    #hydrate model
    model = pickle.loads(model_s)

    #make predictions
    predict_data_b = get_request_data()
    if type(predict_data_b) == type('string'):
        return flask.Response(response = predict_data_b,
                          status = 415,
                          mimetype = 'text/plain')

    predict_data = bytes_to_df(predict_data_b)
    predict_data = preprocess_data(predict_data)

    to_return = {}
    to_return['predictions'] = model.predict(predict_data).tolist()
    try:
        to_return['prediction_probability'] = model.predict_proba(predict_data).tolist()
    except AttributeError:
        pass

    return flask.Response(response = json.dumps(to_return),
                          status = 202,
                          mimetype = 'text/plain')

################### Smaller Helper Functions with no Business Logic ########################

def check_task(task_id):
    res = celery.AsyncResult(task_id)
    if res.state == states.PENDING:
        state = "Model is Training"
    else:
        state = "Ready to Use"
    to_return = {'status': state}

    return flask.Response(response = json.dumps(to_return),
                          status = 200,
                          mimetype = 'text/plain')

def get_request_data():
    if flask.request.content_type == 'text/csv':
        return flask.request.data

    if flask.request.method == 'POST': #training expects label in final column
        return 'Please post data in csv format, and make sure the label column is the last column in the csv.'
    return 'Please post data in csv format.'

def bytes_to_df(data_b):
    if type(data_b) == type(b'byte'):
        data_b = data_b.decode('utf-8')
    data_s = StringIO(data_b)
    data_df = pd.read_csv(data_s, header = None)

    return data_df

if __name__ == '__main__':
    app.run(debug = True)

