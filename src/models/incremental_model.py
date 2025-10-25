from river import forest, metrics, preprocessing, compose
import pickle
import mlflow
import mlflow.pyfunc
import numpy as np

class IncrementalFraudDetector:
    def __init__(self, n_models=10, seed=42):
        self.model = forest.ARFClassifier(
            n_models=n_models,
            seed=seed,
            warning_detector=None,  # Disable concept drift detection for now
            drift_detector=None
        )
        
        # Preprocessing pipeline
        self.preprocessor = compose.Pipeline(
            preprocessing.StandardScaler(),
        )
        
        # Metrics tracking
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        
        self.training_samples = 0
        
    def partial_fit(self, X, y):
        """Incrementally train the model"""
        for i in range(len(X)):
            x_i = X.iloc[i].to_dict()
            y_i = y.iloc[i]
            
            # Make prediction first (for online evaluation)
            if self.training_samples > 0:
                y_pred = self.model.predict_one(x_i)
                
                # Update metrics
                self.accuracy.update(y_i, y_pred)
                self.precision.update(y_i, y_pred)
                self.recall.update(y_i, y_pred)
                self.f1.update(y_i, y_pred)
            
            # Learn from this sample
            self.model.learn_one(x_i, y_i)
            self.training_samples += 1
    
    def predict_proba(self, X):
        """Predict probabilities for batch of samples"""
        probas = []
        for i in range(len(X)):
            x_i = X.iloc[i].to_dict()
            proba = self.model.predict_proba_one(x_i)
            # River returns dict, convert to array [prob_class_0, prob_class_1]
            probas.append([proba.get(0, 0), proba.get(1, 0)])
        return np.array(probas)
    
    def predict(self, X):
        """Predict classes for batch of samples"""
        predictions = []
        for i in range(len(X)):
            x_i = X.iloc[i].to_dict()
            pred = self.model.predict_one(x_i)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_metrics(self):
        """Get current performance metrics"""
        return {
            'accuracy': self.accuracy.get(),
            'precision': self.precision.get(),
            'recall': self.recall.get(),
            'f1': self.f1.get(),
            'training_samples': self.training_samples
        }
    
    def save_model(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# MLflow wrapper for the incremental model
class MLflowIncrementalModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]  # Return fraud probability