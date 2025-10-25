#!/bin/bash

echo "Starting continuous training loop..."

# 🚩 Change following items based on your setup
source .mlops/bin/activate


while true; do
    echo "Checking for new data..."
    
    # Generate new batch (🚩 simulate new data arrival)
    python src/data/generate_new_batch.py
    
    # Run continuous training
    python src/models/simple_continuous_train.py
    
    echo "Sleeping for 1 minutes..."
    sleep 60 # 1 minutes
done
