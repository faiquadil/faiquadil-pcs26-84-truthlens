
from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "phase1" / "ml_selected_pipeline.joblib"

# Make custom project modules available when loading the saved model
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load trained model
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
label_map = bundle["label_map"]

# Convert numeric predictions back to labels
reverse_label_map = {int(value): key for key, value in label_map.items()}

app = FastAPI(
    title="TruthLens AI API",
    description="Fake News Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsRequest(BaseModel):
    title: str = Field(default="", max_length=1000)
    text: str = Field(default="", max_length=50000)

    @model_validator(mode="after")
    def validate_content(self):
        self.title = self.title.strip()
        self.text = self.text.strip()

        if not self.title and not self.text:
            raise ValueError("Please provide a headline or article text.")

        return self


@app.get("/")
def home():
    return {
        "message": "TruthLens AI Backend is running",
        "model": bundle.get("variant", "Selected model")
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_news(request: NewsRequest):
    try:
        input_data = pd.DataFrame([{
            "title": request.title,
            "text": request.text
        }])

        prediction = int(pipeline.predict(input_data)[0])
        label = reverse_label_map.get(prediction, str(prediction))

        return {
            "prediction": label,
            "label": label,
            "raw_prediction": prediction,
            "model": bundle.get("variant", "Selected model"),
            "disclaimer": bundle.get(
                "disclaimer",
                "This prediction is an automated model assessment, not proof of authenticity."
            )
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed. Please try again."
        )
