from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import subprocess

app = FastAPI()

class ScoreRequest(BaseModel):
    session_id: str

@app.post("/process/")
async def process_aria_data(session_id: str):
    # Call Aria Studio CLI or your processing function
    result = subprocess.run(["./aria_process.sh", session_id], capture_output=True, text=True)
    return {"status": result.returncode, "output": result.stdout}

@app.post("/score/")
async def score_driver(data: ScoreRequest):
    # Run your scoring function
    score = your_scoring_function(data.session_id)
    return {"score": score}
