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
MODEL_PATH = "app/models/intent_recognition/xlmr"

# Initialize model components to None
tokenizer = None
model = None
id_to_label = None

try:
    # Check if the model directory exists and contains expected files
    model_config_path = os.path.join(MODEL_PATH, "config.json")
    model_weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin") # Or model.safetensors
    label_map_path = os.path.join(MODEL_PATH, "label_map.json")

    if not os.path.exists(MODEL_PATH) or \
       not os.path.isdir(MODEL_PATH) or \
       not os.path.exists(model_config_path) or \
       not (os.path.exists(model_weights_path) or os.path.exists(os.path.join(MODEL_PATH, "model.safetensors"))) or \
       not os.path.exists(label_map_path):
        
        logger.error(f"Intent model files not found or incomplete in '{MODEL_PATH}'. "
                     f"Please ensure the model is trained and saved correctly to this location.")
        # Raise an exception to prevent further loading attempts with incomplete files
        raise FileNotFoundError(f"Missing intent model files in {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set model to evaluation mode

    # Load label map
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    
    # Invert label map for prediction
    id_to_label = {int(k): v for k, v in label_map.items()}
    logger.info(f"Successfully loaded intent model from {MODEL_PATH}")

except FileNotFoundError as fnfe:
    logger.error(f"Intent Model Loading Error: {fnfe}")
except Exception as e:
    logger.error(f"Error loading intent model or tokenizer during startup: {e}")
    # The variables remain None as initialized, which will trigger the HTTPException later

class TextInput(BaseModel):
    text: str

@router.post("/check", summary="Check intent of a given text")
async def check_intent(text_input: TextInput):
    """
    Predicts the intent of the provided text using the trained model.
    """
    # This check now relies on the outcome of the startup loading process
    if tokenizer is None or model is None or id_to_label is None:
        raise HTTPException(status_code=500, detail="Intent model not loaded. Please train the model and restart the application.")

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
