import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class LGGDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        self.samples = []

        for patient in os.listdir(root_dir):
            patient_dir = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for f in os.listdir(patient_dir):
                if f.endswith(".tif") and not f.endswith("_mask.tif"):
                    img_path = os.path.join(patient_dir, f)
                    mask_path = img_path.replace(".tif", "_mask.tif")
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

        if limit:
            self.samples = self.samples[:limit]

        if len(self.samples) == 0:
            raise RuntimeError("No image/mask pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

        # --- Fix for channel dimension ---
        if img.ndim == 2:  # grayscale image H x W
            img = np.expand_dims(img, axis=0)  # 1 x H x W
        elif img.ndim == 3:  # RGB H x W x C
            img = img.transpose(2, 0, 1)      # C x H x W

        if mask.ndim == 2:  # single channel mask
            mask = np.expand_dims(mask, axis=0)  # 1 x H x W

        # Convert to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask

