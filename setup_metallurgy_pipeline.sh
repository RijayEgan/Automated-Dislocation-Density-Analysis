#!/usr/bin/env zsh
set -euo pipefail

echo "Creating 5‑phase Metallurgy dislocation‑density pipeline..."

# -----------------------------
# Phase 1 — Gold Set Annotation UI (Streamlit)
# -----------------------------
mkdir -p Metallurgy_Phase_1/images

cat > Metallurgy_Phase_1/app.py <<'PY'
#!/usr/bin/env python3
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Dislocation Gold Labeling", layout="wide")

IMG_DIR = Path("Metallurgy_Phase_1/images")
OUT_JSON = Path("Metallurgy_Phase_1/annotations.json")

if OUT_JSON.exists():
    db = json.loads(OUT_JSON.read_text())
else:
    db = {}

st.title("Metallurgy Phase 1 — Dislocation Gold Labeling UI")
st.write("Draw polylines over dislocation lines. Enter material + scale bar.")

image_files = sorted(list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg")))

if not image_files:
    st.warning(f"Place TEM images in {IMG_DIR}")
    st.stop()

idx = st.number_input("Image index", 0, len(image_files) - 1, 0)
img_path = image_files[idx]
image = Image.open(img_path).convert("RGB")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"Image: {img_path.name}")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.0)",
        stroke_width=2,
        stroke_color="#00ff00",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="polyline",
        key=f"canvas_{img_path.name}",
    )

with col2:
    material = st.text_input("Material (e.g., 'Al-Mg_Alloy')", value="Al-Mg_Alloy")
    scale_bar_nm = st.number_input("Scale bar length (nm)", min_value=1.0, value=200.0)
    scale_bar_px = st.number_input("Scale bar length (pixels)", min_value=1.0, value=100.0)

    nm_per_px = scale_bar_nm / scale_bar_px
    st.markdown("**Derived scale:** nm per pixel")
    st.code(f"{nm_per_px:.4f} nm/px")

    if st.button("Save Annotation"):
        shapes = canvas_result.json_data["objects"] if canvas_result.json_data else []
        polylines = []
        total_px_length = 0.0

        for obj in shapes:
            if obj.get("type") == "path":
                path = obj.get("path", [])
                coords = []
                for cmd in path:
                    if cmd[0] in ("M", "L"):
                        coords.append((cmd[1], cmd[2]))
                if len(coords) >= 2:
                    polylines.append({"type": "polyline", "coords": coords})
                    arr = np.array(coords)
                    seg_len = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1)).sum()
                    total_px_length += float(seg_len)

        total_nm = total_px_length * nm_per_px
        dislocation_count = len(polylines)
        # Placeholder: you can replace with area‑normalized density if you want
        rho = total_nm

        db[str(img_path.name)] = {
            "image_id": img_path.name,
            "material": material,
            "dislocation_count": dislocation_count,
            "total_line_length_nm": total_nm,
            "calculated_density_rho": rho,
            "scale_bar_nm": scale_bar_nm,
            "scale_bar_px": scale_bar_px,
            "annotations": polylines,
            "timestamp": time.time(),
        }

        OUT_JSON.write_text(json.dumps(db, indent=2))
        st.success(f"Saved annotation for {img_path.name}")
PY

# -----------------------------
# Phase 2 — JSON Validation & Density Summary
# -----------------------------
mkdir -p Metallurgy_Phase_2

cat > Metallurgy_Phase_2/validate_and_summarize.py <<'PY'
#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

ANN_PATH = Path("Metallurgy_Phase_1/annotations.json")

def load_annotations():
    if not ANN_PATH.exists():
        raise FileNotFoundError(f"No annotations file at {ANN_PATH}")
    return json.loads(ANN_PATH.read_text())

def validate_structure(data):
    errors = []
    for k, v in data.items():
        required = ["image_id", "material", "dislocation_count",
                    "total_line_length_nm", "calculated_density_rho", "annotations"]
        for r in required:
            if r not in v:
                errors.append((k, f"Missing key: {r}"))
        if not isinstance(v.get("annotations", []), list):
            errors.append((k, "annotations is not a list"))
    return errors

def summarize_density(data):
    rhos = [v["calculated_density_rho"] for v in data.values()]
    arr = np.array(rhos, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_rho": 0.0, "std_rho": 0.0}
    return {
        "n": int(arr.size),
        "mean_rho": float(arr.mean()),
        "std_rho": float(arr.std()),
    }

if __name__ == "__main__":
    try:
        data = load_annotations()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    errors = validate_structure(data)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(" -", e)
    else:
        print("JSON structure valid.")

    stats = summarize_density(data)
    print("\nDislocation density summary:")
    for k, v in stats.items():
        print(f"{k}: {v}")
PY

# -----------------------------
# Phase 3 — Teacher Labeling (Multimodal Model)
# -----------------------------
mkdir -p Metallurgy_Phase_3/images

cat > Metallurgy_Phase_3/teacher_labeler.py <<'PY'
#!/usr/bin/env python3
import base64
import json
import os
import time
from pathlib import Path

from anthropic import Anthropic  # swap to Gemini client if preferred

PROMPT = """
You are a crystallographer analyzing a Bright-Field TEM image.
Identify all linear dislocation features. Ignore thickness fringes, bend contours, and moiré patterns.
Return a JSON list of objects with start and end coordinates:

[
  {"x1": 120, "y1": 45, "x2": 180, "y2": 60},
  ...
]

Only return valid JSON. Do not include explanations.
"""

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
IMG_DIR = Path("Metallurgy_Phase_3/images")
OUT = Path("Metallurgy_Phase_3/ai_dislocation_labels.json")

def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def call_model(img_path: Path) -> str:
    img_b64 = encode_image(img_path)
    resp = client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64
                }},
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    return resp.content[0].text.strip()

def main():
    results = {}
    imgs = sorted(list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg")))
    if not imgs:
        print(f"No images found in {IMG_DIR}")
        return

    for img in imgs:
        print("Processing", img.name)
        try:
            txt = call_model(img)
            # try to parse JSON; if it fails, keep raw
            try:
                parsed = json.loads(txt)
            except json.JSONDecodeError:
                parsed = None
            results[img.name] = {"raw_response": txt, "parsed": parsed, "error": None}
        except Exception as e:
            results[img.name] = {"raw_response": None, "parsed": None, "error": str(e)}
        time.sleep(0.2)

    OUT.write_text(json.dumps(results, indent=2))
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
PY

# -----------------------------
# Phase 4 — U-Net Student Training
# -----------------------------
mkdir -p Metallurgy_Phase_4

cat > Metallurgy_Phase_4/train_unet.py <<'PY'
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
PY

# -----------------------------
# Phase 5 — QC & Inference Helper
# -----------------------------
mkdir -p Metallurgy_Phase_5

cat > Metallurgy_Phase_5/infer_and_qc.py <<'PY'
#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from skimage.morphology import skeletonize

from Metallurgy_Phase_4.train_unet import UNetSmall  # reuse model definition

MODEL_PATH = Path("Metallurgy_Phase_4/student_model/unet_student.pt")
GOLD_JSON = Path("Metallurgy_Phase_1/annotations.json")

def load_gold_stats():
    if not GOLD_JSON.exists():
        return None
    data = json.loads(GOLD_JSON.read_text())
    rhos = [v["calculated_density_rho"] for v in data.values()]
    arr = np.array(rhos, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean()), float(arr.std())

def load_model(device):
    model = UNetSmall().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_mask(model, img_path, size=256, device="cpu"):
    img = Image.open(img_path).convert("L")
    orig_w, orig_h = img.size
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs > 0.5).astype(np.uint8)
    # resize back to original size
    mask_img = Image.fromarray(mask * 255).resize((orig_w, orig_h), Image.NEAREST)
    mask_arr = (np.array(mask_img) > 127).astype(np.uint8)
    return mask_arr

def compute_length_nm(mask, scale_bar_nm=200.0, scale_bar_px=100.0):
    nm_per_px = scale_bar_nm / scale_bar_px
    skel = skeletonize(mask > 0)
    total_px = skel.sum()
    total_nm = total_px * nm_per_px
    return total_nm, skel

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run student model on a TEM image and compute density.")
    parser.add_argument("image", type=str, help="Path to TEM image")
    parser.add_argument("--scale_nm", type=float, default=200.0, help="Scale bar length in nm")
    parser.add_argument("--scale_px", type=float, default=100.0, help="Scale bar length in pixels")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print("Image not found:", img_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH)
        return

    model = load_model(device)
    mask = predict_mask(model, img_path, device=device)
    total_nm, skel = compute_length_nm(mask, args.scale_nm, args.scale_px)

    # area in nm^2 (approximate, using scale bar)
    nm_per_px = args.scale_nm / args.scale_px
    h, w = mask.shape
    area_nm2 = (w * nm_per_px) * (h * nm_per_px)
    rho = total_nm / area_nm2 if area_nm2 > 0 else 0.0

    gold_stats = load_gold_stats()
    if gold_stats is not None:
        mean_rho, std_rho = gold_stats
        deviation_pct = 100.0 * (rho - mean_rho) / mean_rho if mean_rho != 0 else 0.0
    else:
        deviation_pct = 0.0

    print(f"Image: {img_path.name}")
    print(f"Total line length: {total_nm:.2f} nm")
    print(f"Estimated dislocation density rho: {rho:.3e} (1/nm)")
    print(f"Deviation from gold mean: {deviation_pct:.1f}%")

    if abs(deviation_pct) > 10.0:
        print("WARNING: Density deviates >10% from gold set mean. Flag for review.")

if __name__ == "__main__":
    main()
PY

echo "All Metallurgy phases created successfully."
