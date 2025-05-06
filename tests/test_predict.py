import sys
import pytest
from PIL import Image
import numpy as np
import torch
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import ChangeDetectionPredictor


# Dummy subclass that overrides model with identity-like mock - that only returns a tensor
#of shape same as the output image with all ones 
class DummyPredictor(ChangeDetectionPredictor):
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = lambda x1, x2: torch.ones((1, 1, 256, 256))  # fixed output
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        from torchvision import transforms as T
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        
#to similate input - returns a solid RGB image of given size
def create_dummy_image(size=(512, 512), color=(255, 0, 0)):
    return Image.new("RGB", size, color)

# Test 1 : 
def test_predict_output_shape():
    predictor = DummyPredictor()
    before_img = create_dummy_image()
    after_img = create_dummy_image()

    change_map = predictor.predict(before_img, after_img)
    #t1 : verifying that the output of predict() is a NumPy array
    assert isinstance(change_map, np.ndarray)
    #t2 : matches the shape of the resized input
    assert change_map.shape == (512, 512)
    #t3 : All values are either 1 or 255 (binary-like output) from the fixed dummy model
    assert (change_map == 1.0).all() or (change_map == 255).all()  # binary-like

def test_real_predict_output_range():
    predictor = ChangeDetectionPredictor("cdSiamese_model_best.pth")
    before = Image.open(r"D:\Sem-6\Software Engineering\Project\Codes\beforeBuilding.png")
    after = Image.open(r"D:\Sem-6\Software Engineering\Project\Codes\afterBuilding.png")



    output = predictor.predict(before, after)
    assert output.shape == (256, 256)

    assert output.min() >= 0.0
    assert output.max() <= 1.0
    
#test 4 : to test non-image input  
#This test checks whether the predict method correctly raises 
# a ValueError when invalid input types are passed for before_image and after_image.

#The first test ("not_an_image") ensures that an invalid string as before_image triggers 
# the appropriate error with the message containing "before_image".
def test_predict_with_non_image_input():
    predictor = DummyPredictor()

    with pytest.raises(ValueError, match="before_image"):
        predictor.predict("not_an_image", create_dummy_image())

    with pytest.raises(ValueError, match="after_image"):
        #The match="after_image" argument ensures that the test will 
        # pass if the error message contains "after_image", which it should.
        predictor.predict(create_dummy_image(), 12345)


#test2 : to check if grayscale images (mode - 'L') is converted to rgb before processing
def test_predict_with_grayscale_input():
    predictor = DummyPredictor()
    gray_image = Image.new("L", (512, 512), 128)  # grayscale

    change_map = predictor.predict(gray_image, gray_image)
    assert change_map.shape == (512, 512) 

#test 3 : if there is a mismatch of input sizes btw before and after, is that handled automatically
#confirming that the prediction succeeds even with inconsistent input sizes
def test_predict_with_mismatched_sizes():
    predictor = DummyPredictor()
    img1 = create_dummy_image((512, 512))
    img2 = create_dummy_image((256, 256))  # different size

    result = predictor.predict(img1, img2)
    assert result.shape == (512, 512)  # auto-resized in preprocess
