import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects

"""
Purpose: segments mri slices using threshold-based segmentation.
Uses a very simple algorithm for proof of concept, can modify later as needed to improve segmentation. 

Inputs: image (type: np.ndarray
Output: image mask (the segmentation) (type: np.float32) """ 

# Classical threshold-based segmentation
def segment_mri_slice(img: np.ndarray) -> np.ndarray:
thresh = threshold_otsu(img)
mask = img > thresh
mask = binary_closing(mask)
mask = remove_small_objects(mask, min_size=100)
return mask.astype(np.float32)