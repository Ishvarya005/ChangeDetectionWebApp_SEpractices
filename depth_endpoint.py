import torch
import numpy as np
import io
import base64
import os
from PIL import Image
import torchvision.transforms as transforms
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List
import uuid

# Create a router for the depth endpoints
router = APIRouter()

# Templates setup
templates = Jinja2Templates(directory="templates")

# Create temporary directory for storing depth images if it doesn't exist
DEPTH_TEMP_DIR = "temp_depth_images"
if not os.path.exists(DEPTH_TEMP_DIR):
    os.makedirs(DEPTH_TEMP_DIR)

# Function to load MiDaS depth estimation model
def load_depth_model():
    """Load MiDaS depth estimation model from local file."""
    try:
        # Create the model structure (we still need to use torch.hub for the architecture)
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, skip_validation=True)
        
        # Now load the weights from the local file
        model_path = "midas_v21_small_256.pt"
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Successfully loaded MiDaS model from: {model_path}")
    except Exception as e:
        print(f"Failed to load local model: {e}")
        raise e  # Re-raise the exception to fail if the model can't be loaded
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Use custom transforms instead of the default MiDaS transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, device

# Initialize depth estimation model 
depth_model, depth_transform, device = load_depth_model()

def estimate_depth(image):
    """Estimate depth for a single image."""
    # Convert PIL image to tensor using our transform
    input_tensor = depth_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = depth_model(input_tensor)
        
        # Resize to original image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
    
    return normalized_depth, depth_map

def create_colored_depth_map(depth_map):
    """Create a colored visualization of the depth map."""
    # Create a colormap (using viridis: deep=purple, shallow=yellow)
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    cmap = LinearSegmentedColormap.from_list('depth_cmap', colors)
    
    # Apply colormap
    colored_depth = (cmap(depth_map) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    colored_depth_img = Image.fromarray(colored_depth[:, :, :3])
    return colored_depth_img

def encode_image_to_base64(image):
    """Convert PIL Image to base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

@router.post("/depth_maps")
async def generate_depth_maps(before: UploadFile = File(...), after: UploadFile = File(...)):
    """Generate depth maps for before and after images (single pair)."""
    # Open images
    before_image = Image.open(before.file).convert("RGB")
    after_image = Image.open(after.file).convert("RGB")
    
    # Estimate depth for both images
    before_depth_norm, before_depth_raw = estimate_depth(before_image)
    after_depth_norm, after_depth_raw = estimate_depth(after_image)
    
    # Calculate depth difference
    depth_diff = np.abs(after_depth_raw - before_depth_raw)
    depth_diff_norm = depth_diff / depth_diff.max()  # Normalize for visualization
    
    # Create colored visualizations
    before_colored = create_colored_depth_map(before_depth_norm)
    after_colored = create_colored_depth_map(after_depth_norm)
    diff_colored = create_colored_depth_map(depth_diff_norm)
    
    # Convert to base64 for sending to frontend
    before_depth_b64 = encode_image_to_base64(before_colored)
    after_depth_b64 = encode_image_to_base64(after_colored)
    diff_depth_b64 = encode_image_to_base64(diff_colored)
    
    # Return all depth maps
    return JSONResponse({
        "before_depth": before_depth_b64,
        "after_depth": after_depth_b64,
        "depth_difference": diff_depth_b64
    })

@router.post("/depth_maps_multiple", response_class=HTMLResponse)
async def generate_multiple_depth_maps(
    request: Request,
    before_images: List[UploadFile] = File(...),
    after_images: List[UploadFile] = File(...)
):
    """Generate depth maps for multiple before and after image pairs."""
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
        # Create a unique ID for this image pair
        pair_id = str(uuid.uuid4())
        
        # Open images
        before_image = Image.open(before_file.file).convert("RGB")
        after_image = Image.open(after_file.file).convert("RGB")
        
        # Save original images
        before_path = f"{DEPTH_TEMP_DIR}/before_{pair_id}.png"
        after_path = f"{DEPTH_TEMP_DIR}/after_{pair_id}.png"
        before_image.save(before_path)
        after_image.save(after_path)
        
        # Estimate depth for both images
        before_depth_norm, before_depth_raw = estimate_depth(before_image)
        after_depth_norm, after_depth_raw = estimate_depth(after_image)
        
        # Calculate depth difference
        depth_diff = np.abs(after_depth_raw - before_depth_raw)
        depth_diff_norm = depth_diff / depth_diff.max()  # Normalize for visualization
        
        # Create colored visualizations
        before_colored = create_colored_depth_map(before_depth_norm)
        after_colored = create_colored_depth_map(after_depth_norm)
        diff_colored = create_colored_depth_map(depth_diff_norm)
        
        # Save depth maps
        before_depth_path = f"{DEPTH_TEMP_DIR}/before_depth_{pair_id}.png"
        after_depth_path = f"{DEPTH_TEMP_DIR}/after_depth_{pair_id}.png"
        diff_depth_path = f"{DEPTH_TEMP_DIR}/diff_depth_{pair_id}.png"
        
        before_colored.save(before_depth_path)
        after_colored.save(after_depth_path)
        diff_colored.save(diff_depth_path)
        
        # Create base64 versions for embedding in HTML
        with open(before_path, "rb") as img_file:
            before_b64 = base64.b64encode(img_file.read()).decode()
        
        with open(after_path, "rb") as img_file:
            after_b64 = base64.b64encode(img_file.read()).decode()
            
        with open(before_depth_path, "rb") as img_file:
            before_depth_b64 = base64.b64encode(img_file.read()).decode()
            
        with open(after_depth_path, "rb") as img_file:
            after_depth_b64 = base64.b64encode(img_file.read()).decode()
            
        with open(diff_depth_path, "rb") as img_file:
            diff_depth_b64 = base64.b64encode(img_file.read()).decode()
        
        # Add to results
        results.append({
            "id": pair_id,
            "before_name": before_file.filename,
            "after_name": after_file.filename,
            "before_img": f"data:image/png;base64,{before_b64}",
            "after_img": f"data:image/png;base64,{after_b64}",
            "before_depth": f"data:image/png;base64,{before_depth_b64}",
            "after_depth": f"data:image/png;base64,{after_depth_b64}",
            "depth_difference": f"data:image/png;base64,{diff_depth_b64}"
        })
    
    # Return template with results
    return templates.TemplateResponse(
        "depth_results.html", 
        {"request": request, "results": results}
    )

# API endpoint that returns JSON for programmatic access
@router.post("/api/depth_maps_multiple")
async def api_generate_multiple_depth_maps(
    before_images: List[UploadFile] = File(...),
    after_images: List[UploadFile] = File(...)
):
    """API endpoint to generate depth maps for multiple image pairs and return JSON."""
    # Validate input
    if len(before_images) != len(after_images):
        return JSONResponse(
            status_code=400,
            content={"error": "Number of before and after images must match"}
        )
    
    if len(before_images) > 5:
        return JSONResponse(
            status_code=400,
            content={"error": "Maximum 5 image pairs allowed"}
        )
    
    results = []
    
    for i, (before_file, after_file) in enumerate(zip(before_images, after_images)):
        # Open images
        before_image = Image.open(before_file.file).convert("RGB")
        after_image = Image.open(after_file.file).convert("RGB")
        
        # Estimate depth for both images
        before_depth_norm, before_depth_raw = estimate_depth(before_image)
        after_depth_norm, after_depth_raw = estimate_depth(after_image)
        
        # Calculate depth difference
        depth_diff = np.abs(after_depth_raw - before_depth_raw)
        depth_diff_norm = depth_diff / depth_diff.max()
        
        # Create colored visualizations
        before_colored = create_colored_depth_map(before_depth_norm)
        after_colored = create_colored_depth_map(after_depth_norm)
        diff_colored = create_colored_depth_map(depth_diff_norm)
        
        # Convert to base64
        before_depth_b64 = encode_image_to_base64(before_colored)
        after_depth_b64 = encode_image_to_base64(after_colored)
        diff_depth_b64 = encode_image_to_base64(diff_colored)
        
        # Add to results
        results.append({
            "pair_index": i + 1,
            "before_name": before_file.filename,
            "after_name": after_file.filename,
            "before_depth": before_depth_b64,
            "after_depth": after_depth_b64,
            "depth_difference": diff_depth_b64
        })
    
    return JSONResponse(content={"results": results})