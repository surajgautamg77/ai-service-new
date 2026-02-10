from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from app.api.intentdetection_xlmr import train_model, check_intent
from app.api.entityextraction_xlm_r_ner import check_ner

app = FastAPI(title="ai service new", version="1.0")

app.include_router(train_model.router, prefix="/api/v1/training", tags=["training"])
app.include_router(check_intent.router, prefix="/api/v1/intent", tags=["intent"])
app.include_router(check_ner.router, prefix="/api/v1/ner", tags=["ner"])

# Set up Jinja2Templates for serving HTML
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Hello FastAPI!"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)