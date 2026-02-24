from ultralytics import YOLO
import torch

# =========================
# VÉRIFICATIONS ENVIRONNEMENT
# =========================
assert torch.cuda.is_available(), "CUDA non disponible"
print("GPU :", torch.cuda.get_device_name(0))
print("CUDA PyTorch :", torch.version.cuda)

# =========================
# PARAMÈTRES
# =========================
DATASET_YAML = "dataset.yaml"   # chemin vers votre dataset.yaml
MODEL = "yolo26s.pt"

IMGSZ = 128
EPOCHS = 300
BATCH = 32

# =========================
# CHARGEMENT MODÈLE
# =========================
model = YOLO(MODEL)

# =========================
# ENTRAÎNEMENT
# =========================
model.train(
    data=DATASET_YAML,
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    patience=25,

    # # Augmentations géométriques adaptées 128x128
    # mosaic=0.0,
    # mixup=0.0,
    # degrees=5.0,      # rotations ±5°
    # scale=0.2,
    # translate=0.05,
    # fliplr=0.5,

    # # Augmentations photométriques (HSV)
    # hsv_h=0.015,
    # hsv_s=0.5,
    # hsv_v=0.4,

    # # Désactivations explicites
    # shear=0.0,
    # perspective=0.0,
    # flipud=0.0,
    # erasing=0.0,

    # Stabilité / performance
    workers=8,
    cache=True,
    device=0
)


print("Entraînement terminé")
