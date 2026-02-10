from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
import logging
import os
import tempfile
from app.training.train_ner_model import train_ner_model as run_actual_ner_training

router = APIRouter()
logger = logging.getLogger(__name__)

def run_ner_training_in_background(file_path: str):
    """
    Runs the actual NER model training logic using the provided CSV file.
    """
    try:
        logger.info(f"Starting actual NER training with CSV from {file_path}...")
        run_actual_ner_training(file_path)
        logger.info(f"Finished actual NER training with CSV from {file_path}.")
    except Exception as e:
        logger.error(f"Error during NER training: {e}", exc_info=True)
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary NER training file: {file_path}")

@router.post("/train", summary="Trigger NER model training with a CSV file")
async def trigger_ner_model_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file containing training data (text and labels columns in BIO/IOB2 format)")
):
    """
    Triggers the training process for an Entity Recognition model using an uploaded CSV file.
    The training runs as a background task.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Create a temporary file to store the uploaded CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_file_path = tmp_file.name
    
    logger.info(f"Received NER training request with file: {file.filename}. Saved to temporary path: {temp_file_path}")
    
    background_tasks.add_task(
        run_ner_training_in_background,
        temp_file_path
    )
    
    return {"message": "NER Training started in background", "filename": file.filename}
