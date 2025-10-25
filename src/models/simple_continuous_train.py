import pandas as pd
import glob
import sys
import os
sys.path.append('src')
from models.simple_model import SimpleFraudDetector
from sklearn.preprocessing import LabelEncoder

def main():
    # Load existing model
    try:
        model = SimpleFraudDetector.load_model('models/simple_fraud_detector.pkl')
        print("Loaded existing model")
    except Exception as e:
        print(f"No existing model found: {e}")
        return
    
    # Find new data batches
    new_files = glob.glob('data/new_batches/*.csv')
    if not new_files:
        print("No new data found")
        return
    
    print(f"Found {len(new_files)} new batches")
    
    # Initialize LabelEncoder and fit with known classes
    le = LabelEncoder()
    known_classes = ['gas', 'grocery', 'online', 'restaurant', 'retail']
    le.fit(known_classes)  # Fit the encoder with known classes
    
    total_new_samples = 0
    for file in new_files:
        print(f"Processing {file}")
        data = pd.read_csv(file)
        
        # Preprocess
        try:
            data['merchant_category'] = le.transform(data['merchant_category'])
        except ValueError as e:
            print(f"Error in transforming merchant_category: {e}")
            print("Ensure all merchant_category values are in", known_classes)
            continue
        
        # Features and target
        feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 'previous_amount', 'time_since_last']
        X = data[feature_cols]
        y = data['is_fraud']
        
        # Continue training
        model.fit(X, y)
        total_new_samples += len(data)
        
        # Move processed file
        os.makedirs('data/processed_batches', exist_ok=True)
        os.rename(file, f'data/processed_batches/{os.path.basename(file)}')
    
    # Save updated model
    model.save_model('models/simple_fraud_detector.pkl')
    print(f"Model updated with {total_new_samples} new samples")
    print(f"Total training samples: {model.training_samples}")

if __name__ == "__main__":
    main()