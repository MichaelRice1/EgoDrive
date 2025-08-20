

cd MSc_AI_Thesis
source .venv/bin/activate

# Full project path
PROJECT_DIR="notebooks_and_scripts/dashboard"
cd "$PROJECT_DIR"

# Start Streamlit using full path to dash.py
streamlit run "$PROJECT_DIR/dash.py"