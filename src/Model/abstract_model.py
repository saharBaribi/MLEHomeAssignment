from abc import ABC
import pandas as pd
from typing import Optional


class AbstractModel(ABC):
    data: pd.DataFrame
    training_data: Optional[tuple]
    validation_data: Optional[tuple]
    test_data: Optional[tuple]
    model_path: str

    def split_data(self) -> tuple:
        pass

    def train_model(self):
        pass

    def evaluate(self):
        pass

    def predict(self) -> str:
        pass

    def calculate_approval_rate(self):
        pass
