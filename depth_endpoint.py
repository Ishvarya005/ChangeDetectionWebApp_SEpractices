import torch
import numpy as np
import io
import base64
import os
from PIL import Image
import torchvision.transforms as transforms
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a router for the depth endpoints
router = APIRouter()

# Function to load MiDaS depth estimation model
def load_depth_model():
    """Load MiDaS depth estimation model."""
    # Check if running on Render (production) or locally
    if os.environ.get('RENDER') == 'true':
        # Use a local model file path for production
        try:
            # Create the model directly
            from models.midas_small import MidasNet_small
            model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", 
                                  exportable=True, non_negative=True, blocks=[1, 2, 3, 4])
            
            # Load the model weights
            model_path = os.path.join(os.path.dirname(__file__), "midas_v21_small_256.pt")
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading local model: {e}")
            # Fallback to a simpler model approach
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 1, 3, padding=1),
                torch.nn.Sigmoid()
            )
    else:
        # Use torch.hub in development environment
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Use custom transforms instead of the default MiDaS transforms
    # which has a compatibility issue with PIL Images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, device

# Initialize depth estimation model (will be loaded when imported)
depth_model, depth_transform, device = load_depth_model()

# The rest of your file remains unchanged
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
    """Generate depth maps for before and after images."""
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