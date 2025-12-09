import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), #input=3 for rgb
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.dec = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.enc(x)
        return self.dec(x)
