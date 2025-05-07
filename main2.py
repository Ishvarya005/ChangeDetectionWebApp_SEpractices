from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import os
import base64
from typing import List
import uuid
from predict import ChangeDetectionPredictor
from depth_endpoint import router as depth_router

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TEMPLATES & STATIC FILES
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create temporary directory for storing images if it doesn't exist
TEMP_DIR = "temp_images"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Set RENDER flag
os.environ['RENDER'] = 'true'

# Load the model
MODEL_PATH = "cdSiamese_model_best.pth"
predictor = ChangeDetectionPredictor(MODEL_PATH)

# Include API routers
app.include_router(depth_router, prefix="/api")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def serve_model_ui(request: Request):
    return templates.TemplateResponse("detect&change.html", {"request": request})

@app.post("/predict")
async def predict_change_map(before: UploadFile = File(...), after: UploadFile = File(...)):
    """Single image pair prediction endpoint for backward compatibility"""
    before_image = Image.open(before.file).convert("RGB")
    after_image = Image.open(after.file).convert("RGB")

    change_map = predictor.predict(before_image, after_image)
    binary_map = (change_map > 0.5).astype('uint8') * 255
    result_image = Image.fromarray(binary_map)
    result_image = result_image.resize(before_image.size, Image.NEAREST)

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/predict-multiple", response_class=HTMLResponse)
async def predict_multiple_images(
    request: Request,
    before_images: List[UploadFile] = File(...),
    after_images: List[UploadFile] = File(...)
):
    # Validate input
    if len(before_images) != len(after_images):
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": "Number of before and after images must match"}
        )
    
    if len(before_images) > 5:
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": "Maximum 5 image pairs allowed"}
        )
    
    results = []
    
    for i, (before_file, after_file) in enumerate(zip(before_images, after_images)):
        # Process each image pair
        before_image = Image.open(before_file.file).convert("RGB")
        after_image = Image.open(after_file.file).convert("RGB")
        
        # Generate a unique ID for this image pair
        pair_id = str(uuid.uuid4())
        
        # Save original images to display
        before_path = f"{TEMP_DIR}/before_{pair_id}.png"
        after_path = f"{TEMP_DIR}/after_{pair_id}.png"
        before_image.save(before_path)
        after_image.save(after_path)
        
        # Run prediction
        change_map = predictor.predict(before_image, after_image)
        binary_map = (change_map > 0.5).astype('uint8') * 255
        result_image = Image.fromarray(binary_map)
        result_image = result_image.resize(before_image.size, Image.NEAREST)
        
        # Save result image
        result_path = f"{TEMP_DIR}/result_{pair_id}.png"
        result_image.save(result_path)
        
        # Create base64 versions for embedding in HTML
        with open(before_path, "rb") as img_file:
            before_b64 = base64.b64encode(img_file.read()).decode()
        
        with open(after_path, "rb") as img_file:
            after_b64 = base64.b64encode(img_file.read()).decode()
            
        with open(result_path, "rb") as img_file:
            result_b64 = base64.b64encode(img_file.read()).decode()
        
        # Add to results
        results.append({
            "id": pair_id,
            "before_name": before_file.filename,
            "after_name": after_file.filename,
            "before_img": f"data:image/png;base64,{before_b64}",
            "after_img": f"data:image/png;base64,{after_b64}",
            "result_img": f"data:image/png;base64,{result_b64}"
        })
    
    # Return template with results
    return templates.TemplateResponse(
        "results.html", 
        {"request": request, "results": results}
    )

# Cleanup function to delete temp files periodically (could be added)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)