import torch
import torch.nn as nn
from torchvision import models, transforms as T
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Dict, Union, Optional

# Model definition
class ChangeDetectionUNet(nn.Module):
    def __init__(self):
        super(ChangeDetectionUNet, self).__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Identity()

        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x1, x2):
        f1 = self.encoder.layer4(self.encoder.layer3(self.encoder.layer2(
            self.encoder.layer1(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x1)))))))
        f2 = self.encoder.layer4(self.encoder.layer3(self.encoder.layer2(
            self.encoder.layer1(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x2)))))))

        x = torch.cat([f1, f2], dim=1)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x

class ChangeDetectionPredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = ChangeDetectionUNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess_image(self, image):
        """Preprocess a single image for the model."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

    def predict(self, before_image, after_image, output_path=None):
        """Process a single pair of before/after images."""
        # Input validation
        if not isinstance(before_image, Image.Image):
            raise ValueError("Expected 'before_image' to be a PIL.Image.Image object")
        if not isinstance(after_image, Image.Image):
            raise ValueError("Expected 'after_image' to be a PIL.Image.Image object")
            
        with torch.no_grad():
            before_tensor = self.preprocess_image(before_image).unsqueeze(0).to(self.device)
            after_tensor = self.preprocess_image(after_image).unsqueeze(0).to(self.device)

            output = self.model(before_tensor, after_tensor)
            change_map = output[0, 0].cpu().numpy()

            # Threshold at 0.5 to get binary change map
            binary_map = (change_map > 0.5).astype(np.uint8) * 255
            change_image = Image.fromarray(binary_map)

            if output_path:
                change_image.save(output_path)
                print(f"[✔] Change map saved to: {output_path}")

            return change_map
    
    def predict_batch(self, before_images: List[Image.Image], after_images: List[Image.Image], 
                      output_paths: Optional[List[str]] = None) -> List[np.ndarray]:
        """Process multiple pairs of before/after images efficiently.
        
        Args:
            before_images: List of before images (PIL Image objects)
            after_images: List of after images (PIL Image objects)
            output_paths: Optional list of paths to save results
            
        Returns:
            List of change maps (numpy arrays)
        """
        if len(before_images) != len(after_images):
            raise ValueError("Number of before and after images must match")
            
        # If output_paths provided, make sure it matches the input length
        if output_paths is not None and len(output_paths) != len(before_images):
            raise ValueError("If output_paths is provided, it must match the number of image pairs")
        
        # Default output_paths to None if not provided
        if output_paths is None:
            output_paths = [None] * len(before_images)
        
        results = []
        batch_size = 2  # Process in small batches to avoid memory issues
        
        # Process in batches
        for i in range(0, len(before_images), batch_size):
            batch_before_images = before_images[i:i+batch_size]
            batch_after_images = after_images[i:i+batch_size]
            batch_paths = output_paths[i:i+batch_size]
            
            # Process this batch
            with torch.no_grad():
                # Preprocess and stack images into a batch
                before_tensors = torch.stack([
                    self.preprocess_image(img) for img in batch_before_images
                ]).to(self.device)
                
                after_tensors = torch.stack([
                    self.preprocess_image(img) for img in batch_after_images
                ]).to(self.device)
                
                # Run prediction
                outputs = self.model(before_tensors, after_tensors)
                
                # Process each result in the batch
                for j, output in enumerate(outputs):
                    change_map = output[0].cpu().numpy()
                    
                    # Save if path is provided
                    if batch_paths[j]:
                        binary_map = (change_map > 0.5).astype(np.uint8) * 255
                        change_image = Image.fromarray(binary_map)
                        change_image.save(batch_paths[j])
                        print(f"[✔] Change map saved to: {batch_paths[j]}")
                    
                    results.append(change_map)
        
        return results
    
    def predict_multiple(self, before_images: List[Image.Image], after_images: List[Image.Image], 
                        temp_dir: str = "temp_change_maps") -> List[Dict[str, Union[str, np.ndarray, Image.Image]]]:
        """Process multiple pairs of before/after images and return detailed results.
        
        Args:
            before_images: List of before images (PIL Image objects)
            after_images: List of after images (PIL Image objects)
            temp_dir: Directory to save temporary files
            
        Returns:
            List of dictionaries containing:
                - change_map: Raw change map as numpy array
                - binary_map: Binary change map as numpy array
                - change_image: Binary change map as PIL Image
                - change_percentage: Percentage of changed pixels
        """
        if len(before_images) != len(after_images):
            raise ValueError("Number of before and after images must match")
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        results = []
        
        # Process image pairs one by one
        for i, (before_img, after_img) in enumerate(zip(before_images, after_images)):
            # Original sizes for resizing back
            original_size = before_img.size
            
            # Generate a filename for this pair
            output_path = os.path.join(temp_dir, f"change_map_{i}.png")
            
            # Run prediction
            change_map = self.predict(before_img, after_img, output_path)
            
            # Calculate binary map
            binary_map = (change_map > 0.5).astype(np.uint8) * 255
            change_image = Image.fromarray(binary_map)
            
            # Resize to original image dimensions
            change_image_resized = change_image.resize(original_size, Image.NEAREST)
            
            # Calculate change percentage
            change_percentage = (binary_map > 0).sum() / binary_map.size * 100
            
            # Add to results
            results.append({
                "change_map": change_map,
                "binary_map": binary_map,
                "change_image": change_image_resized,
                "change_percentage": change_percentage,
                "output_path": output_path
            })
        
        return results

    def predict_from_paths(self, before_path, after_path, output_path=None):
        """Process a single pair of images from file paths."""
        before_image = Image.open(before_path)
        after_image = Image.open(after_path)
        return self.predict(before_image, after_image, output_path)
    
    def predict_multiple_from_paths(self, before_paths: List[str], after_paths: List[str], 
                                   output_dir: str = "output_change_maps") -> List[Dict]:
        """Process multiple pairs of images from file paths.
        
        Args:
            before_paths: List of paths to before images
            after_paths: List of paths to after images
            output_dir: Directory to save output images
            
        Returns:
            List of dictionaries with results
        """
        if len(before_paths) != len(after_paths):
            raise ValueError("Number of before and after image paths must match")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all images
        before_images = [Image.open(path).convert("RGB") for path in before_paths]
        after_images = [Image.open(path).convert("RGB") for path in after_paths]
        
        # Generate output paths
        output_paths = [os.path.join(output_dir, f"change_map_{i}.png") for i in range(len(before_paths))]
        
        # Run batch prediction
        results = self.predict_multiple(before_images, after_images, output_dir)
        
        # Add filenames to results
        for i, result in enumerate(results):
            result["before_path"] = before_paths[i]
            result["after_path"] = after_paths[i]
            
        return results

# Example usage
if __name__ == "__main__":
    predictor = ChangeDetectionPredictor("cdSiamese_model_best.pth")

    # Single prediction example
    change_map = predictor.predict_from_paths(
        "test/A/train_638.png",
        "test/B/train_638.png",
        "change_map_binary.png"
    )

    print(f"Change map shape: {change_map.shape}")
    print(f"Changed pixels (prob > 0.5): {(change_map > 0.5).sum()}")
    
    # Multiple prediction example
    before_paths = ["test/A/train_638.png", "test/A/train_639.png", "test/A/train_640.png"]
    after_paths = ["test/B/train_638.png", "test/B/train_639.png", "test/B/train_640.png"]
    
    results = predictor.predict_multiple_from_paths(before_paths, after_paths)
    
    for i, result in enumerate(results):
        print(f"Image pair {i+1}:")
        print(f"  Change percentage: {result['change_percentage']:.2f}%")
        print(f"  Output saved to: {result['output_path']}")