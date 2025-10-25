from river import forest
import pickle
import numpy as np

"""
🚩 forest.ARFClassifier:

Adaptive Random Forest (ARF) classifier has incremental learning properties.
ARF is specifically designed for data stream mining and can update its model
incrementally as new data arrives.
"""

class SimpleFraudDetector:
    def __init__(self):
        self.model = forest.ARFClassifier(n_models=5, seed=42)
        self.training_samples = 0
        
    def fit(self, X, y):
        for i in range(len(X)):
            x_i = X.iloc[i].to_dict()
            y_i = y.iloc[i]
            self.model.learn_one(x_i, y_i)
            self.training_samples += 1
    
    def predict_proba(self, X):
        probas = []
        for i in range(len(X)):
            x_i = X.iloc[i].to_dict()
            proba = self.model.predict_proba_one(x_i)
            probas.append([proba.get(0, 0), proba.get(1, 0)])
        return np.array(probas)
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)