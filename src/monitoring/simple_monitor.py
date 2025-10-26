from prometheus_client import Counter

class SimpleMonitor:
    def __init__(self):
        self.prediction_counter = Counter('predictions_total', 'Total predictions 🚀')
        self.fraud_predictions = Counter('fraud_predictions_total', 'Fraud predictions 😟')
   
      
    def log_prediction(self, is_fraud):
        self.prediction_counter.inc()
        if is_fraud:
            self.fraud_predictions.inc()


monitor = SimpleMonitor()