# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import os
import random
import numpy as np

# -----------------------------
# Paths
# -----------------------------
input_folder = "Matricaire_fil_rouge-main/Matricaire_fil_rouge-main/pipeline_yolo/blabla/"
output_folder = "Matricaire_fil_rouge-main/Matricaire_fil_rouge-main/pipeline_yolo/blabla_predictions/"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Model
# -----------------------------
model = YOLO(
    "Matricaire_fil_rouge-main/Matricaire_fil_rouge-main/pipeline_yolo/train_yolov8/runs/detect/sol1_medium/weights/best.pt"
)

# -----------------------------
# Tiling parameters
# -----------------------------
TILE_SIZE = 128
OVERLAP = 0
STRIDE = int(TILE_SIZE * (1 - OVERLAP))  

CONF_THRES = 0.25

# -----------------------------
# Loop over images
# -----------------------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    H, W = img.shape[:2]
    print(f"\nProcessing {filename} ({W}x{H})")

    # Copy for drawing
    output_img = img.copy()

    total_boxes = 0

    # -----------------------------
    # Slide over image
    # -----------------------------
    for y in range(0, H - TILE_SIZE + 1, STRIDE):
        for x in range(0, W - TILE_SIZE + 1, STRIDE):

            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]

            # Safety check
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue

            # YOLO inference on tile
            results = model(tile, imgsz=128, conf=CONF_THRES, verbose=False, classes=[0])

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to full-image coordinates
                    x1_full = int(x1 + x)
                    y1_full = int(y1 + y)
                    x2_full = int(x2 + x)
                    y2_full = int(y2 + y)

                    # Draw box
                    #color = (
                    #    random.randint(50, 255),
                    #    random.randint(50, 255),
                    #    random.randint(50, 255),
                    #)

                    color = (
                        0,
                        0,
                        255,
                    )

                    cv2.rectangle(
                        output_img,
                        (x1_full, y1_full),
                        (x2_full, y2_full),
                        color,
                        1,
                    )

                    conf = float(box.conf[0])
                    cv2.putText(output_img,f"{conf:.2f}",(x1_full, y1_full - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,)

                    total_boxes += 1

    # -----------------------------
    # Save result
    # -----------------------------
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, output_img)

    print(f"Detected {total_boxes} boxes")

print("\nAll images processed.")