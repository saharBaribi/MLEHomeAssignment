from abc import ABC
import pandas as pd
from typing import Optional
import numpy as np


class AbstractModel(ABC):
    data: pd.DataFrame
    training_data: Optional[tuple]
    validation_data: Optional[tuple]
    test_data: Optional[tuple]
    model_path: str

    def split_data(self):
        pass

    def train_model(self):
        pass

    def evaluate(self, y_pred: np.array, y_true: np.array):
        pass

    def predict(self, x: pd.DataFrame) -> np.array:
        pass

    def calculate_approval_rate(self, y_pred: np.array):
        pass
