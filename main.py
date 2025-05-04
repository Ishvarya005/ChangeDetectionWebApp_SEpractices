from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
from PIL import Image
import uvicorn
import torch
import os

# Import the change detection model
from predict import ChangeDetectionPredictor

app = FastAPI()

# Serve HTML templates from the "templates" folder
templates = Jinja2Templates(directory="templates")

# Optional: Serve static files if needed (e.g., CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the change detection model
# You should update the path to your trained model file
MODEL_PATH = "cdSiamese_model_best.pth"  # Update this path to your model file
predictor = ChangeDetectionPredictor(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_change_map(before: UploadFile = File(...), after: UploadFile = File(...)):
    # Load the before and after images
    before_image = Image.open(before.file).convert("RGB")
    after_image = Image.open(after.file).convert("RGB")

    # Use the change detection model to generate a change map
    change_map = predictor.predict(before_image, after_image)
    
    # Convert the numpy array to a PIL Image (binary change map)
    binary_map = (change_map > 0.5).astype('uint8') * 255
    result_image = Image.fromarray(binary_map)
    
    # You might want to resize the result to match the original image size
    result_image = result_image.resize(before_image.size, Image.NEAREST)

    # Convert result image to bytes for streaming response
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)