from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import sys
from datetime import datetime
import uvicorn
import yaml
import os
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge
from fastapi import Response
sys.path.append('src')
from models.incremental_model import IncrementalFraudDetector

#SimpleMOnitor Class#####################################################
class SimpleMonitor:
    def __init__(self):
        try:
            # Try to create new metrics
            self.prediction_counter = Counter('predictions_total', 'Total predictions 🚀')
            self.fraud_predictions = Counter('fraud_predictions_total', 'Fraud predictions 😟')
            self.correct_predictions = Counter('correct_predictions_total', 'Correct predictions ✅')
            self.total_feedbacks = Counter('total_feedbacks', 'Total feedback received 📊')
            self.accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy 🎯')
        except ValueError:
            print("Metrics already registered, retrieving existing ones...")
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name'):
                    if collector._name == 'predictions_total':
                        self.prediction_counter = collector
                    elif collector._name == 'fraud_predictions_total':
                        self.fraud_predictions = collector
                    elif collector._name == 'correct_predictions_total':
                        self.correct_predictions = collector
                    elif collector._name == 'total_feedbacks':
                        self.total_feedbacks = collector
                    elif collector._name == 'model_accuracy':
                        self.accuracy_gauge = collector
            
            
            if not hasattr(self, 'prediction_counter'):
                import time
                suffix = str(int(time.time()))
                self.prediction_counter = Counter(f'predictions_total_{suffix}', 'Total predictions 🚀')
                self.fraud_predictions = Counter(f'fraud_predictions_total_{suffix}', 'Fraud predictions 😟')
                self.correct_predictions = Counter(f'correct_predictions_total_{suffix}', 'Correct predictions ✅')
                self.total_feedbacks = Counter(f'total_feedbacks_{suffix}', 'Total feedback received 📊')
                self.accuracy_gauge = Gauge(f'model_accuracy_{suffix}', 'Current model accuracy 🎯')
   
    def log_prediction(self, is_fraud):
        self.prediction_counter.inc()
        if is_fraud:
            self.fraud_predictions.inc()
    
    def log_feedback(self, predicted, actual):
        """Log feedback to track accuracy"""
        self.total_feedbacks.inc()
        if predicted == actual:
            self.correct_predictions.inc()
        
        accuracy = self.get_accuracy()
        self.accuracy_gauge.set(accuracy)
    
    def get_accuracy(self):
        total = self.total_feedbacks._value._value
        correct = self.correct_predictions._value._value
        if total == 0:
            return 0.0
        return correct / total

model = None
preprocessor = None
monitor = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, monitor
    
    if monitor is None:
        monitor = SimpleMonitor()
        print("Simple monitor initialized")
    
    
    model_path = 'models/fraud_detector.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    
    print(f"Checking if model file exists: {os.path.exists(model_path)}")
    print(f"Checking if preprocessor file exists: {os.path.exists(preprocessor_path)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of models directory: {os.listdir('models') if os.path.exists('models') else 'models directory not found'}")
    
    try:
        print(f"Attempting to load model from: {model_path}")
        model = IncrementalFraudDetector.load_model(model_path)
        print("Model loaded successfully")
        
        print(f"Attempting to load preprocessor from: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("Preprocessor loaded successfully")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Make sure the model files are in the correct location")
    except Exception as e:
        print(f"Error during initialization: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    yield
    print("Shutting down...")

app = FastAPI(
    title="Fraud Detection API", 
    docs_url=None, 
    redoc_url=None,
    lifespan=lifespan
)

instrumentator = Instrumentator()
instrumentator.instrument(app)

#Two Classes############################################################

class TransactionRequest(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    merchant_category: str
    previous_amount: float
    time_since_last: float

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: float
    timestamp: str

#6 endpoints############################################################
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud probability for a transaction"""
    global model, preprocessor, monitor
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        data = pd.DataFrame([transaction.model_dump()])
        
        processed_data = preprocessor.transform(data)
        
        feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                       'previous_amount', 'time_since_last', 'amount_log', 
                       'is_weekend', 'is_night', 'amount_ratio']
        
        X = processed_data[feature_cols]
        
        fraud_prob = model.predict_proba(X)[0, 1]  
        is_fraud = fraud_prob > 0.5
        confidence = max(fraud_prob, 1 - fraud_prob)
        
        monitor.log_prediction(is_fraud)
        
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            is_fraud=is_fraud,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/feedback")
async def receive_feedback(transaction: TransactionRequest, actual_label: bool):
    global model, preprocessor, monitor
    
    try:
        data = pd.DataFrame([transaction.model_dump()])
        data['is_fraud'] = int(actual_label)
        
        processed_data = preprocessor.transform(data)
        
        feature_cols = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                       'previous_amount', 'time_since_last', 'amount_log', 
                       'is_weekend', 'is_night', 'amount_ratio']
        
        X = processed_data[feature_cols]
        y = processed_data['is_fraud']
        
        predicted_prob = model.predict_proba(X)[0, 1]
        predicted_label = predicted_prob > 0.5
        
        monitor.log_feedback(predicted_label, actual_label)
       
        model.partial_fit(X, y)
        
        return {
            "status": "feedback_received", 
            "current_accuracy": monitor.get_accuracy(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    global model, monitor
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_metrics = model.get_metrics()
    
    return {
        "model_metrics": model_metrics,
        "accuracy": monitor.get_accuracy(),
        "total_predictions": monitor.prediction_counter._value._value,
        "fraud_predictions": monitor.fraud_predictions._value._value,
        "correct_predictions": monitor.correct_predictions._value._value,
        "total_feedbacks": monitor.total_feedbacks._value._value,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/prometheus")
async def get_prometheus_metrics():
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/reload-model")
async def reload_model():
    global model, preprocessor
    try:
        print("Reloading model...")
        model = IncrementalFraudDetector.load_model('models/fraud_detector.pkl')
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("Model reloaded successfully")
        return {
            "status": "model reloaded successfully", 
            "timestamp": datetime.now().isoformat(),
            "new_metrics": model.get_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

#####################################################################
config_path = "params.yaml"

with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

if __name__ == "__main__":
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])