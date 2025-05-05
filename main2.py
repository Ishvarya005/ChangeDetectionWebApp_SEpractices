from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import uvicorn
import torch
import os

# Import the change detection model
from predict import ChangeDetectionPredictor

# Import routers
from depth_endpoint import router as depth_router
# from integrated_viewer_endpoint import router as ply_router

app = FastAPI()
# app.include_router(ply_router, prefix="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML templates from the "templates" folder
templates = Jinja2Templates(directory="templates")

# Serve static files - important to enable access to model files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Tell the app this is running on Render
os.environ['RENDER'] = 'true'

# Initialize the change detection model
MODEL_PATH = "cdSiamese_model_best.pth" 
predictor = ChangeDetectionPredictor(MODEL_PATH)

# Include the routers for depth and pointcloud endpoints
app.include_router(depth_router, prefix="/api")
# app.include_router(pointcloud_router, prefix="/api")

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
    
    # Resize the result to match the original image size
    result_image = result_image.resize(before_image.size, Image.NEAREST)

    # Convert result image to bytes for streaming response
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
#to start the server - uvicorn main2:app --host 0.0.0.0 --port 8000
