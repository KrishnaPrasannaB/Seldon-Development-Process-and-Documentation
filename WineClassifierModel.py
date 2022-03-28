#WineClassifierModel.py
import joblib
import numpy as np

class WineClassifierModel(object):

    def __init__(self):
        self.model = joblib.load('wine-classifier-model.pkl')

    def predict(self, X, feature_names):
        return self.model.predict_proba(X)