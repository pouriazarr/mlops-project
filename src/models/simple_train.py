import pandas as pd
import sys
sys.path.append('src')

"""
Appending 'src' to sys.path in Python is done to explicitly inform
the Python interpreter where to look for modules and packages
when an import statement is encountered.
"""

from models.simple_model import SimpleFraudDetector
from sklearn.preprocessing import LabelEncoder
import os

def main():
    print("Loading data...")
    data = pd.read_csv('data/raw/initial_data.csv')
    
    print("Preprocessing data...")
    le = LabelEncoder()
    data['merchant_category'] = le.fit_transform(data['merchant_category'])
    
    # Features and target
    feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 'previous_amount', 'time_since_last']
    X = data[feature_cols]
    y = data['is_fraud']
    
    # Train model
    print("Training model...")
    model = SimpleFraudDetector()
    model.fit(X, y)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/simple_fraud_detector.pkl')
    
    print(f"Model trained with {model.training_samples} samples")
    print("Model saved to models/simple_fraud_detector.pkl")

if __name__ == "__main__":
    main()