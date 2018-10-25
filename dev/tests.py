import unittest
from mock import patch

from app import get_request_data
class get_request_data_test(unittest.TestCase):
    @patch('app.get_request_data')
    def test_csv_returns_data(self, mock_data):
        pass