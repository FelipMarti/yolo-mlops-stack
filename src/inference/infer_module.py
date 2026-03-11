import os
import numpy as np
import cv2
import requests
import torch
import time
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")

model = YOLO(str(MODEL_PATH))

print(f"Using device: {DEVICE}")
print(f"Running model {MODEL_PATH}")


def load_bytes(image_bytes: bytes):

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError("Invalid image")

    return image


def load_url(url: str):

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return load_bytes(response.content)


def run_inference(image):

    start = time.perf_counter()

    results = model(image, imgsz=1024, device=DEVICE)

    output_data = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):

            x1, y1, x2, y2 = box

            output_data.append({
                "class_id": int(cls),
                "confidence": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })

    elapsed = time.perf_counter() - start

    return output_data, elapsed
