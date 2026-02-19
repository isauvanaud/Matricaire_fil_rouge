import os
import random
import cv2
import numpy as np

# =========================
# PARAMÈTRES
# =========================
BACKGROUND_CLASS_ID = 1
NB_BACKGROUND_BOXES = 5
MU = 15      
SIGMA = 10   
MAX_TRIES = 1000


# =========================
# OUTILS GÉOMÉTRIQUES
# =========================
def yolo_to_xyxy(box, img_w, img_h):
    _, xc, yc, w, h = box
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def boxes_overlap(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xA < xB and yA < yB:
        return True
    return False


# =========================
# TRAITEMENT PRINCIPAL
# =========================
def add_background_boxes(folder_path):
    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            continue

        base = file.replace(".txt", "")
        img_path = os.path.join(folder_path, base + ".png")
        txt_path = os.path.join(folder_path, file)

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # Charger les annotations existantes
        boxes_xyxy = []
        yolo_boxes = []

        with open(txt_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                yolo_boxes.append(parts)
                boxes_xyxy.append(yolo_to_xyxy(parts, img_w, img_h))

        # Génération des boîtes sol
        added = 0
        tries = 0

        while added < NB_BACKGROUND_BOXES and tries < MAX_TRIES:
            tries += 1

            size = int(max(5, np.random.normal(MU, SIGMA)))
            size = min(size, img_w - 1, img_h - 1)

            x1 = random.randint(0, img_w - size)
            y1 = random.randint(0, img_h - size)
            x2 = x1 + size
            y2 = y1 + size

            candidate = [x1, y1, x2, y2]

            if any(boxes_overlap(candidate, b) for b in boxes_xyxy):
                continue

            # Ajouter la boîte
            xc, yc, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            yolo_boxes.append([BACKGROUND_CLASS_ID, xc, yc, w, h])
            boxes_xyxy.append(candidate)
            added += 1

        # Réécriture du fichier
        with open(txt_path, "w") as f:
            for b in yolo_boxes:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        print(f"{file}: {added} boîtes sol ajoutées")


# =========================
# LANCEMENT
# =========================
if __name__ == "__main__":
    dataset_path = "./tot_annot_add_background"
    add_background_boxes(dataset_path)
