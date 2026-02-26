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
