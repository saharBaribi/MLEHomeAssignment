import pandas as pd


def read_data(path: str = "./data/Riskified-MLE-home-assignmnet-data.csv") -> pd.DataFrame:
    """
        We don't necessarily need a function for this, but the function can change to read from other sources
        # that do not exist in this repo like S3 or others
    """
    data = pd.read_csv(path)
    return data
