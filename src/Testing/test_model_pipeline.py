import unittest

from src.Model.preprocessing import Preprocessing
from src.Utils.utils import read_data


class TestModelPipeline(unittest.TestCase):
    def test_preprocess_data(self):
        # Test that the preprocessing function behaves as expected
        raw_data = read_data(path="../../data/Riskified-MLE-home-assignmnet-data.csv")  # Load some test data
        self.assertIsNotNone(raw_data)
        features_list = ["billing_country_code", "shipping_country_code", "shipping_method", "total_spent",
                         "currency_code", "V4_our_age", "V5_merchant_age", "V8_ip"]
        preprocessing = Preprocessing(raw_data, features_list)

        processed_data = preprocessing.run_preprocessing()
        self.assertIsNotNone(processed_data)
        self.assertIn('label', processed_data.columns,
                      "Label column 'label' is missing")


if __name__ == '__main__':
    unittest.main()
