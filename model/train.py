import torch
from torch.utils.data import DataLoader, random_split
from dataset import LGGDataset
from model import TinyUNet
from losses import dice_loss
import kagglehub

# -----------------------
# Dataset
# -----------------------
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
data_root = f"{path}/kaggle_3m"

dataset = LGGDataset(data_root, limit=200)  # keep small
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# -----------------------
# Model
# -----------------------
device = "cpu"
model = TinyUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# Training
# -----------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    dice_scores = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            outputs = model(imgs)
            d = 1 - dice_loss(outputs, masks)
            dice_scores.append(d.item())

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {total_loss/len(train_loader):.4f} | "
        f"Val Dice: {sum(dice_scores)/len(dice_scores):.4f}"
    )

torch.save(model.state_dict(), "tinyunet.pth")
