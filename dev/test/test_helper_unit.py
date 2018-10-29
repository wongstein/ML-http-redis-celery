import unittest
from mock import Mock, patch
from freezegun import freeze_time
import pandas as pd

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