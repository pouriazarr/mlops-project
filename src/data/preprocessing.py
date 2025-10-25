from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.fitted = False
        
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['merchant_category']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df[col])
            
        # Feature engineering
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
        df_processed['amount_ratio'] = df_processed['amount'] / (df_processed['previous_amount'] + 1)
        
        self.fitted = True
        return df_processed
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
            
        df_processed = df.copy()
        
        # Apply same transformations
        for col, encoder in self.label_encoders.items():
            df_processed[col] = encoder.transform(df[col])
            
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
        df_processed['amount_ratio'] = df_processed['amount'] / (df_processed['previous_amount'] + 1)
        
        return df_processed