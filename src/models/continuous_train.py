import os
import glob
import pandas as pd
import mlflow
import json
from datetime import datetime
import yaml
import sys
import requests
sys.path.append('src')
from models.incremental_model import IncrementalFraudDetector
import pickle

def continuous_training(config_path='params.yaml'):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        model = IncrementalFraudDetector.load_model('models/fraud_detector.pkl')
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("Loaded existing model")
    except FileNotFoundError:
        print("No existing model found, please run initial training first")
        return
    
    new_batch_files = glob.glob('data/new_batches/*.csv')
    
    if not new_batch_files:
        print("No new data batches found")
        return
    
    print(f"Found {len(new_batch_files)} new batches")
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow_continious']['experiment_name'])

    
    with mlflow.start_run():
        initial_metrics = model.get_metrics()
        
        total_new_samples = 0
        for batch_file in sorted(new_batch_files):
            print(f"Processing {batch_file}")
            
            new_data = pd.read_csv(batch_file)
            processed_data = preprocessor.transform(new_data)

            feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                           'previous_amount', 'time_since_last', 'amount_log', 
                           'is_weekend', 'is_night', 'amount_ratio']
            
            X_new = processed_data[feature_cols]
            y_new = processed_data['is_fraud']

            model.partial_fit(X_new, y_new)
            total_new_samples += len(X_new)

            processed_dir = 'data/processed_batches'
            os.makedirs(processed_dir, exist_ok=True)
            os.rename(batch_file, os.path.join(processed_dir, os.path.basename(batch_file)))

        final_metrics = model.get_metrics()

        mlflow.log_metric("total_new_samples", total_new_samples)
        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(f"final_{metric_name}", metric_value)

        model.save_model('models/fraud_detector_updated.pkl')

        os.replace('models/fraud_detector_updated.pkl', 'models/fraud_detector.pkl')

        try:
            response = requests.post('http://localhost:8000/reload-model')
            if response.status_code == 200:
                print("✅ Model reloaded in FastAPI successfully")
            else:
                print(f"❌ Failed to reload model: {response.text}")
        except Exception as e:
            print(f"⚠️  Could not reload model automatically: {e}")
            print("💡 Run manually: curl -X POST http://localhost:8000/reload-model")
        
            print(f"Continuous training completed. Processed {total_new_samples} new samples")

        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'new_samples_processed': total_new_samples,
            'initial_training_samples': initial_metrics['training_samples'],
            'final_training_samples': final_metrics['training_samples'],
            'final_accuracy': final_metrics['accuracy'],
            'final_precision': final_metrics['precision'],
            'final_recall': final_metrics['recall'],
            'final_f1': final_metrics['f1']
        }
        
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/continuous_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Continuous training completed. Processed {total_new_samples} new samples")
        print(f"Updated metrics: {final_metrics}")

if __name__ == "__main__":
    continuous_training()