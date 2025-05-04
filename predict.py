import torch
import torch.nn as nn
from torchvision import models, transforms as T
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os

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
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

    def predict(self, before_image, after_image, output_path=None):
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

    def predict_from_paths(self, before_path, after_path, output_path=None):
        before_image = Image.open(before_path)
        after_image = Image.open(after_path)
        return self.predict(before_image, after_image, output_path)

# Example usage
if __name__ == "__main__":
    predictor = ChangeDetectionPredictor("cdSiamese_model_best.pth")

    # Predict and save change map
    change_map = predictor.predict_from_paths(
        r"D:\Sem-6\Computer Vision\Project\LEVIR-CD+\test\A\train_638.png",
        r"D:\Sem-6\Computer Vision\Project\LEVIR-CD+\test\B\train_638.png",
        "change_map_binary.png"
    )

    print(f"Change map shape: {change_map.shape}")
    print(f"Changed pixels (prob > 0.5): {(change_map > 0.2).sum()}")
