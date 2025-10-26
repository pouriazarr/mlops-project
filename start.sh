#!/bin/bash

mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns &

python src/api/app.py