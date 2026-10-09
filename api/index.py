from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import predictor

app = FastAPI(title="Email Spam Classifier API")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class EmailRequest(BaseModel):
    text: str


@app.post("/api/predict")
async def predict_email(request: EmailRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")
    return predictor.predict(request.text)


@app.get("/api/info")
async def model_info():
    return predictor.info()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Must stay last: serves the dashboard at "/"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
