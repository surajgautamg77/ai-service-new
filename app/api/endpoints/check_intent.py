from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

router = APIRouter()
logger = logging.getLogger(__name__)

# Define the model path
MODEL_PATH = "app/models/intent_xlmr"

# Load tokenizer and model once on startup
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set model to evaluation mode

    # Load label map
    with open(os.path.join(MODEL_PATH, "label_map.json"), "r") as f:
        label_map = json.load(f)
    
    # Invert label map for prediction
    id_to_label = {int(k): v for k, v in label_map.items()}

except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    # Depending on the application's tolerance for startup failures,
    # you might want to re-raise the exception or handle it more gracefully.
    # For now, we'll let it proceed, but the endpoint will likely fail.
    tokenizer = None
    model = None
    id_to_label = None

class TextInput(BaseModel):
    text: str

@router.post("/check", summary="Check intent of a given text")
async def check_intent(text_input: TextInput):
    """
    Predicts the intent of the provided text using the trained model.
    """
    if not tokenizer or not model or not id_to_label:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    try:
        inputs = tokenizer(
            text_input.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1) # Calculate probabilities
            predictions = torch.argmax(probabilities, dim=-1) # Get the predicted class

        predicted_label_id = predictions.item()
        predicted_intent = id_to_label.get(predicted_label_id, "Unknown")
        confidence_score = probabilities[0, predicted_label_id].item() # Get confidence score

        return {"text": text_input.text, "predicted_intent": predicted_intent, "confidence_score": confidence_score}

    except Exception as e:
        logger.error(f"Error during intent prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")
