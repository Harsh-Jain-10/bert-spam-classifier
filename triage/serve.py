import os
from pathlib import Path
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from predict_triage import predict_triage, check_model_weights, MODEL_PATH

app = FastAPI(title="Email Triage Multi-Class Classifier Service")

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    scores: dict

@app.on_event("startup")
def startup_event():
    global tokenizer, model, device
    print("Starting up FastAPI email triage classifier service...")
    
    # Pre-check model weights
    if not check_model_weights():
        print("⚠️ ERROR: Model weights file is missing. Please run train_triage.py to train the model first.")
        # We don't raise an exception here so that the app starts, but endpoints will return 503
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BERT model onto device: {device}...")
    
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        print("Model loaded successfully and service is ready.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/health")
def health_check():
    """
    Service health check.
    """
    if model is None or tokenizer is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "healthy", "model": "bert-base-uncased-triage"}

@app.post("/classify", response_model=ClassifyResponse)
def classify_email(request: ClassifyRequest):
    """
    Classify email text into one of the 6 triage classes.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model is not loaded or missing weights. Please train the model."
        )
        
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Request text cannot be empty."
        )
        
    try:
        predicted_label, confidence, scores = predict_triage(
            request.text, tokenizer, model, device
        )
        return ClassifyResponse(
            label=predicted_label,
            confidence=confidence,
            scores=scores
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
