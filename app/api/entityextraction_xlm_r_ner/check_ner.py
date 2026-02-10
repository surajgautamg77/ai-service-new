from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

router = APIRouter()
logger = logging.getLogger(__name__)

# Define the model path for NER
MODEL_PATH = "app/models/entity_extraction/xlmr_ner"
# Define a known pre-trained NER model to fall back to if no local model is found
# User can replace this with their specific fine-tuned NER model ID if hosted on Hugging Face
FALLBACK_PRETRAINED_NER_MODEL = "dslim/bert-base-NER"

# Initialize variables to None
tokenizer = None
model = None
ner_pipeline = None

try:
    # Attempt to load a fine-tuned local model first
    if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH) and \
       os.path.exists(os.path.join(MODEL_PATH, "config.json")) and \
       os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")): # Basic check for model files
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        logger.info(f"Loaded NER model from local path: {MODEL_PATH}")
    else:
        logger.warning(f"Local NER model not found at {MODEL_PATH} or is incomplete. Attempting to load fallback pre-trained NER model: {FALLBACK_PRETRAINED_NER_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_PRETRAINED_NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(FALLBACK_PRETRAINED_NER_MODEL)
        logger.info(f"Loaded fallback pre-trained NER model: {FALLBACK_PRETRAINED_NER_MODEL}")
    
    if tokenizer is not None and model is not None:
        model.eval() # Set model to evaluation mode
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    else:
        raise ValueError("Failed to load both tokenizer and model for NER.")

except Exception as e:
    logger.error(f"Error loading NER model or tokenizer during startup: {e}")

class TextInput(BaseModel):
    text: str

@router.post("/check", summary="Perform Named Entity Recognition (NER) on a given text")
async def check_ner(text_input: TextInput):
    """
    Predicts Named Entities in the provided text using the trained NER model.
    """
    if ner_pipeline is None:
        raise HTTPException(status_code=500, detail="NER model not loaded. Please ensure the model is trained/available and application restarted.")

    try:
        results = ner_pipeline(text_input.text)
        # Convert numpy.float32 scores to standard floats for JSON serialization
        for entity in results:
            if 'score' in entity and isinstance(entity['score'], torch.Tensor):
                entity['score'] = entity['score'].item()
            elif 'score' in entity and hasattr(entity['score'], 'item'): # Handle numpy.float32
                entity['score'] = entity['score'].item()
        return {"text": text_input.text, "entities": results}

    except Exception as e:
        logger.error(f"Error during NER prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during NER prediction: {e}")
