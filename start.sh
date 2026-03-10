#!/bin/bash

echo "Installing torch runtime..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT