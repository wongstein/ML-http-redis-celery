import flask
from flask import Flask, url_for
from worker import celery
import celery.states as states
import redis

from io import StringIO
import json

import pickle
import pandas as pd

from helpers import gen_model_unique_id, gen_model_task_unique_id, preprocess_data, bytes_to_df

app = Flask(__name__)

try:
    r_conn = redis.Redis('redis')
    r_conn.get('test')
except redis.exceptions.ConnectionError:
    # this is a localrun
    r_conn = redis.Redis('localhost')

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

if __name__ == '__main__':
    app.run(debug = True)
