import unittest
from mock import Mock, patch
from freezegun import freeze_time
import pandas as pd
import app import app


class AppFunctionalClass(unittest.TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()

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
        response = self.client.post('/models', content_type = 'not text', data = "['stuff']")
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

        response = self.client.post('/models', content_type = 'text/csv', data = "good data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), '{"message": "You\'re model is being trained right now, you can check it\'s status by hitting the check_model_status endpoint", "model_id": "model_id", "task_id": "task_id"}')

    @patch('app.r_conn')
    def test_predict_404(self, mock_conn):
        mock_conn.get.return_value = None

        response = self.client.get('/models/model_doesntexist')

        self.assertEqual(response.status_code, 404)

    @patch('app.r_conn')
    @patch('app.pickle')
    @patch('app.get_request_data')
    @patch('app.bytes_to_df')
    @patch('app.preprocess_data')

    def test_predict_202(self, mock_preprocess, mock_bytesdf, mock_getreqdata, mock_pickle, mock_conn):
        mock_conn.get.return_value = 'good_model'
        #mock_pickle.loads.return_value = 'A hydrated model'
        mock_getreqdata = b'good data'
        mock_bytesdf.return_value  = 'good_data_df'
        mock_preprocess.return_value = 'processed_data_df'
        mock_pickle.loads.return_value.predict.return_value.tolist.return_value = 'class predictions list'
        mock_pickle.loads.return_value.predict_proba.return_value.tolist.return_value = 'class probability predictions list'


        response = self.client.get('/models/model_exist', content_type = 'text/csv', data = b'good_data')

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data.decode('utf-8'), '{"predictions": "class predictions list", "prediction_probability": "class probability predictions list"}')

