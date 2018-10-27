import unittest
from mock import Mock, patch
from freezegun import freeze_time
import pandas as pd

############## Helper Tests #################################################
from helpers import gen_model_unique_id
@freeze_time("2018-10-01")
class gen_model_unique_id_test(unittest.TestCase):
    def test_gen_unique_id_correct(self):
        expectation = 'model_20181001-000000'
        self.assertEqual(gen_model_unique_id(), expectation)


from helpers import gen_model_task_unique_id
class gen_model_task_unique_id_test(unittest.TestCase):
    def test_gen_unique_id_correct(self):
        expectation = 'model_20181001-000000_task'
        model_id = 'model_20181001-000000'
        self.assertEqual(gen_model_task_unique_id(model_id), expectation)

from helpers import preprocess_data
class PreprocessDataCase(unittest.TestCase):
    @patch('helpers.preprocessing')
    def test_preprocess_data(self, mock_sklearn):
        preprocess_data('some data here')
        mock_sklearn.scale.assert_called_with('some data here')

######## App Unit Tests ###############################################
from app import get_request_data
class GetRequestDataClass(unittest.TestCase):
    @patch('app.flask.request')
    def test_csv_returns_POST(self, mock_flaskrequest):
        mock_flaskrequest.content_type = 'not_csv'
        mock_flaskrequest.method = 'POST'

        returned = get_request_data()
        self.assertEqual(returned, 'Please post data in csv format, and make sure the label column is the last column in the csv.')
    @patch('app.flask.request')
    def test_csv_returns_GET(self, mock_flaskrequest):
        mock_flaskrequest.content_type = 'not_csv'
        mock_flaskrequest.method = 'GET'

        returned = get_request_data()
        self.assertEqual(returned, 'Please post data in csv format.')

    @patch('app.flask.request')
    def test_csv_returns_data(self, mock_flaskrequest):
        mock_flaskrequest.content_type = 'text/csv'
        mock_flaskrequest.data = b'[1,2,3]'

        returned = get_request_data()
        self.assertEqual(returned, b'[1,2,3]')

from app import bytes_to_df
class BytesDFClass(unittest.TestCase):
    def test_new_bytes(self):
        result = bytes_to_df(b'1,2\n2,1')
        expectation = pd.DataFrame([[1,2], [2,1]])

        pd.testing.assert_frame_equal(result, expectation)

from app import check_task
@patch('app.celery')
class CheckTaskClass(unittest.TestCase):
    def test_checktask_calls_dependencies(self, mock_celery):
        check_task('test_task_id')
        mock_celery.AsyncResult.assert_called_with('test_task_id')

from app import check_model_status
class CheckModelStatusClass(unittest.TestCase):
    @patch('app.check_task')
    @patch('app.gen_model_task_unique_id')
    @patch('app.r_conn')
    def test_checkmodel_calls_dependencies_modelexists(self,mock_conn, mock_gen_modelid, mock_checktask):
        mock_gen_modelid.return_value = "some unique model-task id"
        mock_conn.get.return_value = "some task id"
        check_model_status('test_model_id')

        mock_gen_modelid.assert_called_with('test_model_id')
        mock_checktask.assert_called_with("some task id")

    @patch('app.gen_model_task_unique_id')
    @patch('app.r_conn')
    @patch('app.check_task')
    def test_checkmodel_calls_dependencies_modelnoexists(self,mock_checktask, mock_conn, mock_gen_modelid):
        mock_gen_modelid.return_value = None
        mock_conn.get.return_value = None

        check_model_status('test_model_id')

        mock_gen_modelid.assert_called_with('test_model_id')
        self.assertFalse(mock_checktask.called, "Failed to not call check_task when model_id not present in redis")

from app import train_model_endpoint
class TrainModelEndpointClass(unittest.TestCase):
    @patch('app.get_request_data')
    @patch('app.gen_model_unique_id')
    def test_trainmodelendpoint_calls_dependencies_nodata(self, mock_genid, mock_requestdata):
        mock_requestdata.return_value = 'no data here'
        train_model_endpoint()

        mock_requestdata.assert_called_with()
        self.assertFalse(mock_genid.called, "Train model expects request data to be in byte form, else it assumes non-valid data was passed")

    @patch('app.get_request_data')
    @patch('app.celery')
    @patch('app.r_conn')
    @patch('app.gen_model_unique_id')
    @patch('app.gen_model_task_unique_id')
    def test_trainmodelendpoint_calls_dependencies_yesdata(self, mock_genmodeltaskid, mock_genid, mock_conn, mock_celery, mock_requestdata):
        mock_requestdata.return_value = b'good data'
        mock_genid.return_value = "model_id"
        mock_celery.send_task.return_value.id = 'task_id'
        mock_genmodeltaskid.return_value = 'model_task_id'

        train_model_endpoint()

        self.assertEqual(mock_genid.called, True)
        mock_celery.send_task.assert_called_with('tasks.train', args = ['model_id', b'good data'], kwargs = {})
        mock_genmodeltaskid.assert_called_with("model_id")
        mock_conn.set.assert_called_with('model_task_id', 'task_id')

from app import predict
class PredictClass(unittest.TestCase):
    @patch('app.r_conn')
    @patch('app.check_model_status')
    def test_predict_nomodel(self, mock_modelstatus, mock_conn):
        mock_conn.get.return_value = None
        mock_modelstatus.return_value = "Model Doesn't Exist Flask response"
        returned = predict('model_doesntexist')

        self.assertEqual(mock_conn.get.called, True)
        self.assertEqual(returned, "Model Doesn't Exist Flask response")

    @patch('app.r_conn')
    @patch('app.pickle')
    @patch('app.get_request_data')
    @patch('app.bytes_to_df')
    @patch('app.preprocess_data')

    def test_predict_returnspred(self, mock_preprocess, mock_bytesdf, mock_getreqdata, mock_pickle, mock_conn):
        mock_conn.get.return_value = 'good_model'
        mock_getreqdata.return_value = b'good data'
        mock_bytesdf.return_value  = 'good_data_df'
        mock_preprocess.return_value = 'processed_data_df'
        mock_pickle.loads.return_value.predict.return_value.tolist.return_value = 'class predictions list'
        mock_pickle.loads.return_value.predict_proba.return_value.tolist.return_value = 'class probability predictions list'

        predict('good_model')

        mock_pickle.loads.assert_called_with('good_model')
        self.assertTrue(mock_getreqdata.called, True)
        mock_bytesdf.assert_called_with(b'good data')
        mock_preprocess.assert_called_with('good_data_df')
        mock_pickle.loads.return_value.predict.assert_called_with('processed_data_df')
        mock_pickle.loads.return_value.predict_proba.assert_called_with('processed_data_df')

######## Test Celery #####################################################
from app import celery_train_model
class CeleryTrainModelClass(unittest.TestCase):
    @patch('app.pickle')
    @patch('app.bytes_to_df')
    @patch('app.redis')
    def test_celery_train_success(self, mock_redis, mock_bytesdf, mock_pickle):
        mock_pickle.dumps.return_value = 'model in pickle string'
        mock_bytesdf.return_value = pd.DataFrame([[1,2], [2,1]])
        mock_redis.Redis.return_value.set.return_value = True

        result = celery_train_model('model_id', b'[[1,2], [2,1]]')
        mock_bytesdf.assert_called_with(b'[[1,2], [2,1]]')
        self.assertEqual(mock_pickle.dumps.called, True)
        self.assertEqual(mock_redis.Redis.return_value.set.called, True)
        self.assertDictEqual(result, {'status': 'complete', 'unique_id': 'model_id'})

    @patch('app.pickle')
    @patch('app.bytes_to_df')
    @patch('app.redis')
    def test_celery_train_fails(self, mock_redis, mock_bytesdf, mock_pickle):
        mock_pickle.dumps.return_value = 'model in pickle string'
        mock_bytesdf.return_value = pd.DataFrame([[1,2], [2,1]])
        mock_redis.Redis.return_value.set.return_value = False

        result = celery_train_model('model_id', b'[[1,2], [2,1]]')
        mock_bytesdf.assert_called_with(b'[[1,2], [2,1]]')
        self.assertEqual(mock_pickle.dumps.called, True)
        self.assertEqual(mock_redis.Redis.return_value.set.called, True)
        self.assertDictEqual(result, {'status': 'failed', 'unique_id': 'model_id'})

######## Functional App Tests ############################################
import app
from celery_mock import task_mock
class AppFunctionalClass(unittest.TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()

    @patch('app.celery')
    def test_check_task_200_Pending(self, mock_celery):
        mock_celery.AsyncResult.return_value.state = 'PENDING'
        response = self.client.get('check_task_status/test_task_id')

        self.assertEqual(response.data.decode('utf-8'), 'PENDING')
        self.assertEqual(response.status_code, 200)

    @patch('app.celery')
    def test_check_task_200_Finished(self, mock_celery):
        mock_celery.AsyncResult.return_value.result = 'Finished'
        response = self.client.get('check_task_status/test_task_id')

        self.assertEqual(response.data.decode('utf-8'), 'Finished')
        self.assertEqual(response.status_code, 200)


    @patch('app.check_task')
    @patch('app.gen_model_task_unique_id')
    @patch('app.r_conn')
    def test_check_model_404_modelnotfound(self, mock_conn, mock_gen_modelid, mock_checktask):
        mock_gen_modelid.return_value = "test_train_model_id"
        mock_conn.get.return_value = None

        response = self.client.get('check_model_status/test_model')

        self.assertEqual(response.data.decode('utf-8'), "The model id you supplied either doesn't exist or the model_id provided is wrong.")
        self.assertEqual(response.status_code, 404)
        #mock_checktask.assert_called_with('test_model')

    @patch('app.check_task')
    @patch('app.gen_model_task_unique_id')
    @patch('app.r_conn')
    def test_check_model_200_modelstatus(self, mock_conn, mock_gen_modelid, mock_checktask):
        mock_gen_modelid.return_value = "test_train_model_id"
        mock_conn.get.return_value = "some unique id"
        mock_checktask.return_value = "model found and trained"

        response = self.client.get('check_model_status/test_model')

        self.assertEqual(response.data.decode('utf-8'), "model found and trained")
        self.assertEqual(response.status_code, 200)

    def test_trainmodelendpoint_415_baddata(self):
        response = self.client.post('/train', content_type = 'not text', data = "['stuff']")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.data.decode('utf-8'), 'Please post data in csv format, and make sure the label column is the last column in the csv.')

    @patch('app.get_request_data')
    @patch('app.celery')
    @patch('app.r_conn')
    @patch('app.gen_model_unique_id')
    def test_trainmodelendpoint_200(self, mock_genid, mock_conn, mock_celery, mock_requestdata):
        mock_requestdata.return_value = b'good data'
        mock_genid.return_value = "model_id"
        mock_celery.send_task.return_value.id = 'task_id'

        response = self.client.post('/train', content_type = 'text/csv', data = "good data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), '{"message": "You\'re model is being trained right now, you can check it\'s status by hitting the check_task endpoint", "model_id": "model_id", "task_id": "task_id"}')

    @patch('app.r_conn')
    def test_predict_404(self, mock_conn):
        mock_conn.get.return_value = None

        response = self.client.get('/predict/model_doesntexist')

        self.assertEqual(response.status_code, 404)

    @patch('app.r_conn')
    @patch('app.pickle')
    @patch('app.get_request_data')
    @patch('app.bytes_to_df')
    @patch('app.preprocess_data')

    def test_predict_200(self, mock_preprocess, mock_bytesdf, mock_getreqdata, mock_pickle, mock_conn):
        mock_conn.get.return_value = 'good_model'
        #mock_pickle.loads.return_value = 'A hydrated model'
        mock_getreqdata = b'good data'
        mock_bytesdf.return_value  = 'good_data_df'
        mock_preprocess.return_value = 'processed_data_df'
        mock_pickle.loads.return_value.predict.return_value.tolist.return_value = 'class predictions list'
        mock_pickle.loads.return_value.predict_proba.return_value.tolist.return_value = 'class probability predictions list'


        response = self.client.get('/predict/model_exist', content_type = 'text/csv', data = b'good_data')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), '{"predictions": "class predictions list", "prediction_probability": "class probability predictions list"}')


if __name__ == '__main__':
    unittest.main()