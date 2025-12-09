# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import LGGDataset
from model import TinyUNet
from losses import dice_loss

# -------------------------
# Config
# -------------------------
DATA_ROOT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2/kaggle_3m"
)
FP32_MODEL_PATH = "tinyunet.pth"
INT8_MODEL_PATH = "tinyunet_int8.pth"

N_RUNS = 5
BATCH_SIZE = 1
LIMIT = 50   # small subset for eval
DEVICE = "cpu"

OUT_DIR = "quant_results"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Dice metric (thresholded)
# -------------------------
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

# -------------------------
# Seed control
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# -------------------------
# Evaluation loop
# -------------------------
def evaluate(model, loader):
    model.eval()
    dice_scores = []
    samples = []

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(img)
            dice = dice_score(torch.sigmoid(logits), mask)

            dice_scores.append(dice.item())
            samples.append((img.cpu(), mask.cpu(), torch.sigmoid(logits).cpu(), dice.item()))

    return dice_scores, samples

# -------------------------
# Main
# -------------------------
def main():
    dataset = LGGDataset(DATA_ROOT, limit=LIMIT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    fp32_dices_all = []
    int8_dices_all = []

    best_sample = None
    best_dice = -1

    for run in range(N_RUNS):
        print(f"\n=== Run {run + 1}/{N_RUNS} ===")
        set_seed(run)

        fp32 = TinyUNet().to(DEVICE)
        int8 = TinyUNet().to(DEVICE)

        fp32.load_state_dict(torch.load(FP32_MODEL_PATH, map_location=DEVICE))
        int8.load_state_dict(torch.load(INT8_MODEL_PATH, map_location=DEVICE))

        fp32_dices, samples_fp32 = evaluate(fp32, loader)
        int8_dices, samples_int8 = evaluate(int8, loader)

        fp32_dices_all.extend(fp32_dices)
        int8_dices_all.extend(int8_dices)

        # Save best sample (INT8 used)
        for (img, mask, pred, d) in samples_int8:
            if d > best_dice:
                best_dice = d
                best_sample = (img, mask, pred)

    # -------------------------
    # Reporting
    # -------------------------
    def report(name, scores):
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{name}: {mean:.4f} ± {std:.4f}")
        return mean, std

    print("\n=== FINAL RESULTS ===")
    report("FP32 Dice", fp32_dices_all)
    report("INT8 Dice", int8_dices_all)

    # -------------------------
    # Histogram
    # -------------------------
    plt.hist(fp32_dices_all, bins=20, alpha=0.6, label="FP32")
    plt.hist(int8_dices_all, bins=20, alpha=0.6, label="INT8")
    plt.xlabel("Dice Score")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Per-slice Dice Distribution")

    plt.savefig(os.path.join(OUT_DIR, "dice_histogram.png"))
    plt.close()

    print("Saved: dice_histogram.png")

    # -------------------------
    # Qualitative Overlay
    # -------------------------
    img, mask, pred = best_sample

    img = img.squeeze().permute(1, 2, 0)
    mask = mask.squeeze()
    pred = pred.squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Input MRI")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred > 0.5, cmap="gray")
    axs[2].set_title(f"INT8 Prediction (Dice={best_dice:.3f})")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "best_overlay.png"))
    plt.close()

    print("Saved: best_overlay.png")

        # -------------------------
    # Qualitative Overlay (positive cases)
    # -------------------------
    best_sample = None
    best_dice = -1

    # Filter for positive slices in INT8 evaluation
    for img, mask, pred, d in samples_int8:
        if mask.sum() > 0:  # only consider slices with actual tumor
            if d > best_dice:
                best_dice = d
                best_sample = (img, mask, pred)

    if best_sample is None:
        print("No positive slices found. Using original best_sample.")
        # fallback to original best (may be empty)
        best_sample = (samples_int8[0][0], samples_int8[0][1], samples_int8[0][2])
        best_dice = samples_int8[0][3]

    # Convert for plotting
    img, mask, pred = best_sample
    img = img.squeeze().permute(1, 2, 0)
    mask = mask.squeeze()
    pred = pred.squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Input MRI")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred > 0.5, cmap="gray")
    axs[2].set_title(f"INT8 Prediction (Dice={best_dice:.3f})")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "best_positive_overlay.png"))
    plt.close()

    print("Saved: best_positive_overlay.png")


if __name__ == "__main__":
    main()


