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
