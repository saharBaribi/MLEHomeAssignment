import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from typing import Optional
import pandas as pd
import pickle
import logging

from src.Model.abstract_model import AbstractModel


class RandomForest(AbstractModel):
    def __init__(self, data: pd.DataFrame, model_path: str = "./src/Model/TrainedModel/", estimators: int = 100):
        self.data: pd.DataFrame = data
        self.training_data: Optional[tuple] = None
        self.validation_data: Optional[tuple] = None
        self.test_data: Optional[tuple] = None
        self.estimators: int = estimators
        self.model_path: str = model_path

    def run_classifier(self):
        """
        Full classifier flow. Is only needed when training and evaluating the model.
        In cases where only prediction is needed we will only use predict.
        """
        self.split_data()
        self.train_model()
        # Evaluate on validation set:
        y_val_predict = self.predict(x=self.validation_data[0])
        self.evaluate(y_pred=y_val_predict, y_true=self.validation_data[1])
        # Evaluate on test set:
        y_predict = self.predict(x=self.test_data[0])
        self.evaluate(y_pred=y_predict, y_true=self.test_data[1])
        self.calculate_approval_rate(y_pred=y_predict)

    def split_data(self):
        """
        Splitting the data into train, validation and test sets.
        """
        logging.info("Splitting data to train, validation and test")
        X: pd.DataFrame = self.data.drop('label', axis=1)
        y: pd.Series = self.data['label']
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                          stratify=y_train_val, random_state=42)
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def train_model(self):
        logging.info("Training model")
        rf_classifier = RandomForestClassifier(n_estimators=self.estimators, random_state=42)
        rf_classifier.fit(*self.training_data)
        self.save_model(rf_classifier)

    def save_model(self, model: RandomForestClassifier):
        """
            The function saves the trained model to file.
            Usually, we will hold an ML metadata store, and each model will be saved separately, and versioned.
            e.g. we will mot overwrite models, so we will be able to roll back to previous models if necessary
        """
        logging.info("Saving trained model")
        with open(f'{self.model_path}/model.pkl', 'wb') as file:
            pickle.dump(model, file)

    def evaluate(self, y_pred: np.array, y_true: np.array):
        """
        Checking accuracy and classification report. Other metrics can be added.
        We can also save the accuracy overtime to monitor the model performance and trigger retraining if needed.
        """
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", class_report)

    def load_model(self) -> RandomForestClassifier:
        """
        Loading the trained model object. Will happen each time we need to predict.
        """
        try:
            with open(f'{self.model_path}/model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
                return loaded_model
        except:
            logging.error("There is no model is the provided path")
            sys.exit(1)

    def predict(self, x: pd.DataFrame) -> np.array:
        model = self.load_model()
        y_pred = model.predict(x)
        return y_pred

    def calculate_approval_rate(self, y_pred: np.array):
        approval_rate = sum(y_pred) / len(y_pred)
        print(f"The approval rate for the model is: {approval_rate}")
