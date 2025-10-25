import pandas as pd
import numpy as np
from datetime import datetime
import os

class FraudDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_batch(self, n_samples=1000, fraud_rate=0.05):
        data = []
        
        for _ in range(n_samples):
            amount = np.random.lognormal(3, 1.5)
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            merchant_category = np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail'])
            
            
            is_fraud = np.random.random() < fraud_rate
            
            if is_fraud:
                amount *= np.random.uniform(2, 5)
                hour = np.random.choice([2, 3, 4, 22, 23, 0, 1])
                
            
            previous_amount = np.random.lognormal(2.5, 1)
            time_since_last = np.random.exponential(2)
            
            data.append({
                'amount': amount,
                'hour': hour,
                'day_of_week': day_of_week,
                'merchant_category': merchant_category,
                'previous_amount': previous_amount,
                'time_since_last': time_since_last,
                'timestamp': datetime.now(),
                'is_fraud': int(is_fraud)
            })
            
        return pd.DataFrame(data)
    
    def save_batch(self, df, batch_id):
        os.makedirs('data/new_batches', exist_ok=True)
        filepath = f'data/new_batches/batch_{batch_id}.csv'
        df.to_csv(filepath, index=False)
        return filepath