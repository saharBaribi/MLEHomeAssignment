import pandas as pd
from typing import Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class Preprocessing:
    def __init__(self, data: pd.DataFrame, features_list: Optional[list[str]], id_column: str = "order_id",
                 status_column: str = "status"):
        self.data: pd.DataFrame = data
        self.features_for_model: Optional[list[str]] = features_list
        self.id_column: str = id_column
        self.status_column: str = status_column
        self.numeric_cols: Optional[list[str]] = None
        self.categorical_cols: Optional[list[str]] = None

    def create_label(self):
        """
            The label is created based on the following logic -
            if the transaction was approved, the label is 1. Otherwise, we get 0.
            This may be problematic due to the fact the declined transactions have no real label,
             but we don't want to throw 10% of our data away.
            returns : pd dataframe with a label column replacing the status column
        """
        self.data["label"] = self.data["status"].apply(lambda x: 1 if x == "approved" else 0)
        return self.data.drop(["status"], axis=1)

    def filter_features(self):
        # Get only relevant features for training
        self.data = self.data[self.features_for_model + [self.id_column, self.status_column]]

    def run_preprocessing(self) -> pd.DataFrame | None:
        # Select features to run on:
        self.filter_features()
        self.handle_missing_values()
        self.encode_categorial_features()
        preprocessed_data = self.create_label()
        return preprocessed_data.drop(self.id_column, axis=1)

    def handle_missing_values(self):
        # Getting numeric and categorial feature columns
        self.numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        self.numeric_cols.remove(self.id_column)
        self.categorical_cols = list(self.data.select_dtypes(include=['object']).columns)
        self.categorical_cols.remove(self.status_column)

        # Setting up imputer do handle missing values.
        # Numeric column will have the mean value while categorical will get the most freequent.
        imputer_numeric = SimpleImputer(strategy='mean')
        imputer_categorical = SimpleImputer(strategy='most_frequent')

        # Completing missing values in all columns
        self.data[self.numeric_cols] = imputer_numeric.fit_transform(self.data[self.numeric_cols])
        self.data[self.categorical_cols] = imputer_categorical.fit_transform(self.data[self.categorical_cols])

    def encode_categorial_features(self):
        for col in self.categorical_cols:
            if col != 'status':
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
