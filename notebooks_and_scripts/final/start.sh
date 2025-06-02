#!/bin/bash

# Navigate to project directory (change if needed)
cd /Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis

source /Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/.venv/bin/activate

# Full project path
PROJECT_DIR="/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts/final"

cd "$PROJECT_DIR"

# Start FastAPI
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

sleep 2

# Start Streamlit using full path to app.py
streamlit run "$PROJECT_DIR/frontend/app.py"
