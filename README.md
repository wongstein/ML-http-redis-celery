# Project Description
This is a basic, ML logistic regression training/serving http stack using flask to deliver the api, celery and redis to train and temporarily store trained model information.

This api comes with a couple of endpoints that could be useful, and here they are:
### POST /models
* Endpoint description: Training a new logistic regression model.  Post training data,
* Expected fields in post request:
  * Content-Type = 'text/csv'
  * data field contains training data in csv format with no header.
* Response Structure:
  * 201 Success: {"message": "You're model is being trained right now, you can check it's status by hitting the check_task endpoint", "model_id": "some model id", "task_id": "some task id"}
  * 415 Data Format wrong: Please post data in csv format, and make sure the label column is the last column in the csv.'

### GET /models/:model_id
* Endpoint description: Making predictions with data provided in the get request.  This endpoint supports both single predictions and batch predictions.
* Expected fields in post request:
  * Content-Type = 'text/csv'
  * data field contains training data in csv format with no header.
* Response Structure:
  * 202 Accepted: {"predictions": [some class], "prediction_probability": [[some class probabilities...]]}
  * 200 Model Still Training: "Model is Training"
  * 404 Model not Found: "The model id you supplied either doesn't exist or the model_id provided is wrong."
  * 415 Data Format wrong: Please post data in csv format, and make sure the label column is the last column in the csv.'

### POST /check_model_status/:model_id
* Endpoint Description: Check the status of your model using the model id provided in the training endpoint response.
* Response Structure:
  * 200 Model Exists: Response will either be "Model is Training" or "Ready to Use", depending on the state of the model.
  * 404 Model not found: "The model id you supplied either doesn't exist or the model_id provided is wrong."

# Setting up Dev Environment
First, create a new python virtual env using python 3.6.  Then ``` pip install requirements.txt``` to get all your dependencies installed.

## starting redis
```redis-server /usr/local/etc/redis.conf```
check with ```redis-cli ping```.  If you get a "PONG" back, your redis is up and ready to go.

## starting a celery worker
```celery worker -A app.celery --loglevel=INFO```

## starting the flask server
``` python app.py```

# Using the API
## Sending a post request with postman (just an example)

Set a post request to http://localhost:5000/train.  Set content-type in headers to equal 'text/csv'.  In the body, click on raw and set the data type to text.  You can copy and paste the csv input here.  Then push send and watch the magic.

## Making a get request for predict not with postman
Somehow, postman doesn't allow you to attach a body to a get request.  It's okay, you can do it programmatically in your favorite programmatic method.  Here's an example of a python line which will send a get request
```
import requests
model = model_20181025-132027
response = requests.get('http://localhost:5000/predict/%s' % (model), headers = {'Content-Type':'text/csv'}, data = '6.9,3.1,5.1,2.3')
print(response.content)
print(response.status_code)
```

# Running Tests
If you are in the dev folder, you can run:
``` python -m unittest discover ```
