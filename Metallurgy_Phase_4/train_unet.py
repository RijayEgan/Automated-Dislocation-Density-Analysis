#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.morphology import skeletonize

# Simple U-Net-like model (compact)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        m = self.mid(p2)

        u2 = self.up2(m)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        out = self.out_conv(c1)
        return out

class DislocationDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None, size=256):
        self.img_dir = Path(img_dir)
        self.data = json.loads(Path(label_json).read_text())
        self.keys = [k for k, v in self.data.items() if v.get("parsed")]
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.keys)

    def _blank_mask(self, w, h):
        return np.zeros((h, w), dtype=np.uint8)

    def _rasterize_lines(self, w, h, segments):
        mask = self._blank_mask(w, h)
        for seg in segments:
            try:
                x1, y1 = int(seg["x1"]), int(seg["y1"])
                x2, y2 = int(seg["x2"]), int(seg["y2"])
            except Exception:
                continue
            # simple Bresenham-like interpolation
            num = max(abs(x2 - x1), abs(y2 - y1)) + 1
            xs = np.linspace(x1, x2, num).astype(int)
            ys = np.linspace(y1, y2, num).astype(int)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            mask[ys, xs] = 1
        return mask

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.data[key]
        img_path = self.img_dir / key
        img = Image.open(img_path).convert("L")
        w, h = img.size

        segments = entry["parsed"] if isinstance(entry["parsed"], list) else []
        mask = self._rasterize_lines(w, h, segments)

        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask_img = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask_img, dtype=np.float32)

        if self.transform:
            img_tensor = self.transform(img_arr)
        else:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_tensor, mask_tensor

def train():
    img_dir = "Metallurgy_Phase_3/images"
    label_json = "Metallurgy_Phase_3/ai_dislocation_labels.json"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = DislocationDataset(img_dir, label_json, transform=transform, size=256)
    if len(dataset) == 0:
        print("No training data found. Ensure teacher labels and images exist.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    out_dir = Path("Metallurgy_Phase_4/student_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "unet_student.pt")
    print("Saved model to", out_dir / "unet_student.pt")

if __name__ == "__main__":
    train()
