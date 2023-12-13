import pandera as pa
import pandas as pd
from src.Features.transaction_schema import TransactionSchema
from src.Utils.utils import read_data


def test_validate_transaction_data(data: pd.DataFrame = None):
    # Validate the data using the defined schema
    if data is None:
        data = read_data(path="./data/Riskified-MLE-home-assignmnet-data.csv")
    try:
        data = TransactionSchema.validate(data, lazy=True)
        print("Data validation passed.")
        return data
    except pa.errors.SchemaError as e:
        print("Data validation failed:", e)
