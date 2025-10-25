import mlflow
import pickle
import mlflow.pyfunc
import yaml
import sys
import pandas as pd
sys.path.append('src')

from models.incremental_model import IncrementalFraudDetector, MLflowIncrementalModel
from data.preprocessing import DataPreprocessor

def train_model(config_path='params.yaml'):
   
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    # data_config = config['data']
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_config)
        
        # Initialize components
        model = IncrementalFraudDetector(
            n_models=model_config['n_models'],
            seed=model_config['seed']
        )
        
        preprocessor = DataPreprocessor()
        
        initial_data = pd.read_csv("data/raw/initial_data.csv")
        
        # Preprocess data
        processed_data = preprocessor.fit_transform(initial_data)
        
        # Prepare features and target
        feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                       'previous_amount', 'time_since_last', 'amount_log', 
                       'is_weekend', 'is_night', 'amount_ratio']
        
        X = processed_data[feature_cols]
        y = processed_data['is_fraud']
        
        # Initial training
        print(f"Training on {len(X)} samples...")
        model.partial_fit(X, y)
        
        # Log metrics
        metrics = model.get_metrics()
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save model artifacts
        model.save_model('models/fraud_detector.pkl')
        
        # Log model to MLflow
        mlflow_model = MLflowIncrementalModel(model)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=mlflow_model,
            pip_requirements=['river', 'pandas', 'numpy', 'scikit-learn']
        )
        
        # Save preprocessor
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        
        mlflow.log_artifact('models/preprocessor.pkl')
        
        print(f"Model trained. Metrics: {metrics}")
        return model, preprocessor

if __name__ == "__main__":
    train_model()