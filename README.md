# Setting up

First, make sure you have docker installed.  Then, run ``` docker-compose up -d --build ```.


# starting redis
```redis-server /usr/local/etc/redis.conf```
check with ```redis-cli ping```

# starting a celery worker
```celery worker -A app.celery --loglevel=INFO```

# sending a post request with postman (just an example)

Set a post request to http://localhost:5000/train.  Set content-type in headers to equal 'text/csv'.  In the body, click on raw and set the data type to text.  You can copy and paste the csv input here.  Then push send and watch the magic.

# Making a get request for predict not with postman
Somehow, postman doesn't allow you to attach a body to a get request.  It's okay, you can do it programmatically in your favorite programmatic method.  Here's an example of a python line which will send a get request
```
import requests
model = model_20181025-132027
requests.get('http://localhost:5000/predict/%s' % (model), headers = {'Content-Type':'text/csv'}, data = '6.9,3.1,5.1,2.3')
print(requests.content)
print(requests.status_code)
```

