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
