import os
import time
import torch
from model import TinyUNet
from dataset import LGGDataset
from torch.utils.data import DataLoader, random_split
import kagglehub


# -----------------------
# Evaluation
# -----------------------
def evaluate(model, loader):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for imgs, masks in loader:
            preds = model(imgs)
            preds = (preds > 0.5).float()

            intersection = (preds * masks).sum()
            dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-8)

            dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)


# -----------------------
# Main
# -----------------------
def main():
    # Load dataset
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    data_root = f"{path}/kaggle_3m"
    dataset = LGGDataset(data_root, limit=50)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=1)

    # -----------------------
    # Load FP32 model
    # -----------------------
    model_fp32 = TinyUNet()
    model_fp32.load_state_dict(torch.load("tinyunet.pth", map_location="cpu"))
    model_fp32.eval()

    # -----------------------
    # Quantize
    # -----------------------
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Conv2d},
        dtype=torch.qint8
    )

    torch.save(model_int8.state_dict(), "tinyunet_int8.pth")

    # -----------------------
    # Size comparison
    # -----------------------
    fp32_size = os.path.getsize("tinyunet.pth") / 1e6
    int8_size = os.path.getsize("tinyunet_int8.pth") / 1e6

    print(f"FP32 size: {fp32_size:.2f} MB")
    print(f"INT8 size: {int8_size:.2f} MB")

    # -----------------------
    # Speed comparison
    # -----------------------
    dummy_input = torch.randn(1, 3, 256, 256)

    start = time.time()
    for _ in range(20):
        model_fp32(dummy_input)
    fp32_time = time.time() - start

    start = time.time()
    for _ in range(20):
        model_int8(dummy_input)
    int8_time = time.time() - start

    print(f"FP32 inference time: {fp32_time:.3f}s")
    print(f"INT8 inference time: {int8_time:.3f}s")
    print(f"Speedup: {fp32_time / int8_time:.2f}x")

    # -----------------------
    # Dice comparison
    # -----------------------
    fp32_dice = evaluate(model_fp32, val_loader)
    int8_dice = evaluate(model_int8, val_loader)

    print(f"FP32 Dice: {fp32_dice:.4f}")
    print(f"INT8 Dice: {int8_dice:.4f}")


if __name__ == "__main__":
    main()
