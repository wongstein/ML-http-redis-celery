import unittest
from mock import Mock, patch
from freezegun import freeze_time
import pandas as pd

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
