import numpy as np
from PIL import Image


def load_image(path):
img = Image.open(path).convert("L")
return np.array(img).astype(np.float32)