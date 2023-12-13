import logging
import pandas as pd
from src.Model.preprocessing import Preprocessing
from src.Model.random_forest import RandomForest
from src.Testing.test_data_validation import test_validate_transaction_data
from src.Utils.utils import read_data


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data: pd.DataFrame = read_data()
    data: pd.DataFrame = test_validate_transaction_data(data)
    print(len(data))
    #  Usually, we will get the feature list from the user or from a config file. For simplicity, it appears here.
    features_list = ['email_anoni', 'billing_country_code', 'shipping_country_code', 'shipping_method', 'created_at',
                     'total_spent', 'currency_code', 'gateway', 'V1_link', 'V2_distance', 'V3_distance',
                     'V4_our_age', 'V5_merchant_age', 'V6_avs_result', 'V7_bill_ship_name_match', 'V8_ip', 'V9_cookie',
                     'V10_cookie', 'V11_cookie']

    preprocessing = Preprocessing(data=data, features_list=features_list)
    preprocessed_data: pd.DataFrame = preprocessing.run_preprocessing()
    model = RandomForest(data=preprocessed_data)
    model.run_classifier()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
