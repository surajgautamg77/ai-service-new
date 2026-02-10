from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from app.api.endpoints import train_model

app = FastAPI()

app.include_router(train_model.router, prefix="/api/v1/training", tags=["training"])

# Set up Jinja2Templates for serving HTML
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Hello FastAPI!"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)