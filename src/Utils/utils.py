import pandas as pd


def read_data(path: str = "./data/Riskified-MLE-home-assignmnet-data.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    return data
