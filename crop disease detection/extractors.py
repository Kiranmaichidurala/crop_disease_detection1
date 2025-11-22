from PIL import Image
import numpy as np

def extract_features_from_image(image_path, size=(64, 64)):
    """
    Load image, resize to `size`, flatten and return shape (1, n_features) numpy array.
    If your model used other features (histograms, color spaces, etc.) replace accordingly.
    """
    img = Image.open(image_path).convert('RGB').resize(size)
    arr = np.array(img).astype('float32') / 255.0
    return arr.flatten().reshape(1, -1)