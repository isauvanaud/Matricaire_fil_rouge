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

MARGIN_SIZE = 20

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

    H, W = H + TILE_SIZE, W + TILE_SIZE

    black_image = np.zeros((H, W, 3), dtype=np.uint8)
    black_image[int(TILE_SIZE/2):H-int(TILE_SIZE/2), int(TILE_SIZE/2):W-int(TILE_SIZE/2)] = img
    img = black_image


    # Image de traitement pour les masques, image de sortie pour l'affichage
    traitement_img = img.copy()
    output_img = img.copy()

    total_boxes = 0

    coords = [(int(TILE_SIZE/2), H - int(3*TILE_SIZE/2), int(TILE_SIZE/2), W - int(3*TILE_SIZE/2)), #y_min,y_max,x_min,x_max pour run n°1
              (int(TILE_SIZE/2), H - int(3*TILE_SIZE/2), 0               , W - int(TILE_SIZE/2  )), #run n°2
              (0               , H - int(TILE_SIZE/2  ), int(TILE_SIZE/2), W - int(3*TILE_SIZE/2)),
              (0               , H - int(TILE_SIZE/2  ), 0               , W - int(TILE_SIZE/2  )),
             ]

    # -----------------------------
    # Slide over image
    # -----------------------------
    for coord in coords:
        for y in range(coord[0], coord[1], STRIDE):
            for x in range(coord[2], coord[3], STRIDE):

                tile = traitement_img[y:y + TILE_SIZE, x:x + TILE_SIZE]

                results = model(tile, imgsz=128, conf=CONF_THRES, verbose=False, classes=[0])

                x1_safe, y1_safe = MARGIN_SIZE, MARGIN_SIZE
                x2_safe, y2_safe = TILE_SIZE - MARGIN_SIZE, TILE_SIZE - MARGIN_SIZE

                for r in results:
                    if r.boxes is None:
                        continue

                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Traitement frontières
                        # Si la boîte n'est pas dans la zone de confiance du run, on passe immédiatement à l'itération suivante
                        if x2 < x1_safe or y2 < y1_safe or x1 > x2_safe or y1 > y2_safe:
                            continue

                        # Convert to full-image coordinates
                        x1_full = int(x1 + x)
                        y1_full = int(y1 + y)
                        x2_full = int(x2 + x)
                        y2_full = int(y2 + y)

                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cv2.rectangle(traitement_img, pt1=(x1_full,y1_full), pt2=(x2_full,y2_full), color=(0, 0, 0), thickness=-1)

                        # Draw box
                        #color = (
                        #    random.randint(50, 255),
                        #    random.randint(50, 255),
                        #    random.randint(50, 255),
                        #)

                        color = (0,0,255,)

                        cv2.rectangle(output_img,(x1_full, y1_full),(x2_full, y2_full),color,1)

                        #conf = float(box.conf[0])
                        #cv2.putText(output_img,f"{conf:.2f}",(x1_full, y1_full - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,)

                        total_boxes += 1

                cv2.rectangle(traitement_img, pt1=(x+x1_safe,y+y1_safe), pt2=(x+x2_safe,y+y2_safe), color=(0, 0, 0), thickness=-1)

    # -----------------------------
    # Save result
    # -----------------------------
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, output_img)

    print(f"Detected {total_boxes} boxes")

print("\nAll images processed.")
