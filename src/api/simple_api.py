from fastapi import FastAPI
import yaml
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import HTMLResponse
import sys

sys.path.append('src')
from models.simple_model import SimpleFraudDetector
from contextlib import asynccontextmanager
import uvicorn

""" 🚩 uvicorn
Uvicorn is a high-performance ASGI (Asynchronous Server Gateway Interface) server for Python web applications.
Its primary function is to serve as the interface between your asynchronous web framework (like FastAPI or Starlette)
and the underlying server, handling incoming requests and sending back responses.
"""

app = FastAPI(title="Simple Fraud Detection API")

# Global model variable
model = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = SimpleFraudDetector.load_model('models/simple_fraud_detector.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield  
    print("Application shutting down")

app = FastAPI(title="Simple Fraud Detection API", lifespan=lifespan)

class TransactionRequest(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    merchant_category: int 
    previous_amount: float
    time_since_last: float

@app.get("/health")
async def health_check():
    return {"status": "I'm ok BROOOO 😎"}

@app.post("/predict")
async def predict_fraud(transaction: TransactionRequest):
    global model
    if model is None:
        return {"error": "Model not loaded"}
    
    data = pd.DataFrame([transaction.model_dump()])
    
    proba = model.predict_proba(data)[0, 1]
    is_fraud = proba > 0.5
    
    return {
        "fraud_probability": float(proba),
        "is_fraud": bool(is_fraud)
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                }
                .container {
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <p>😎 Welcome to the MLOps Project of this course.</p>
                <p>🚀 Check <a href="/docs">/docs</a> for API documentation.</p>
                <p>😉 Best Wishes for you</p>
                <p><a href="https://m-fozouni.ir/de7" target="_blank" rel="noopener noreferrer">👉 Data Engineering Course</a></p>
            </div>
        </body>
    </html>
    """

# Read From Yaml Configs
config_path = "params.yaml"

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

if __name__ == "__main__":
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])