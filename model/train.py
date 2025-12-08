# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SliceDataset
from model import UNet2D
import numpy as np

DATA_ROOT = "PATH_TO_DATASET"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SliceDataset(DATA_ROOT)
train_ds, val_ds = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = UNet2D().to(device)
loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out, mask)
        loss.backward()
        opt.step()

    model.eval()
    dices = []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = (model(img) > 0.5).float()
            d = (2*(pred*mask).sum())/(pred.sum()+mask.sum()+1e-6)
            dices.append(d.item())

    print(f"Epoch {epoch+1} Dice: {np.mean(dices):.4f}")
