from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
import logging
import os
import tempfile
from app.training.train_intent_model import train_model as run_actual_training

router = APIRouter()
logger = logging.getLogger(__name__)

def run_training_in_background(file_path: str):
    """
    Runs the actual model training logic using the provided CSV file.
    """
    try:
        logger.info(f"Starting actual training with CSV from {file_path}...")
        run_actual_training(file_path)
        logger.info(f"Finished actual training with CSV from {file_path}.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")

@router.post("/train", summary="Trigger model training with a CSV file")
async def trigger_model_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file containing training data (text and intent columns)")
):
    """
    Triggers the training process for an intent detection model using an uploaded CSV file.
    The training runs as a background task.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Create a temporary file to store the uploaded CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_file_path = tmp_file.name
    
    logger.info(f"Received training request with file: {file.filename}. Saved to temporary path: {temp_file_path}")
    
    background_tasks.add_task(
        run_training_in_background,
        temp_file_path
    )
    
    return {"message": "Training started in background", "filename": file.filename}

@router.get("/status", summary="Get training status (placeholder)")
async def get_training_status():
    """
    Placeholder for retrieving the status of ongoing or completed training jobs.
    """
    return {"status": "No active training job found (placeholder)", "last_trained_model": "None"}
