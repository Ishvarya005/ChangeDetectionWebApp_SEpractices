from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import os
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
