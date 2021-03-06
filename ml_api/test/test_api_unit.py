import unittest
from mock import Mock, patch
from freezegun import freeze_time
import pandas as pd

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
        mock_celery.send_task.assert_called_with('tasks.train', args = ['model_id', 'good data'], kwargs = {})
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
