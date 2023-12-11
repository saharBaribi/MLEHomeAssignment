import pandera as pa
import pandas as pd
from src.Features.transaction_schema import TransactionSchema


def test_validate_transaction_data(data: pd.DataFrame):
    # Validate the data using the defined schema
    try:
        data = TransactionSchema.validate(data, lazy=True)
        print("Data validation passed.")
        return data
    except pa.errors.SchemaError as e:
        print("Data validation failed:", e)
