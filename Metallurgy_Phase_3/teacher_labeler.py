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
