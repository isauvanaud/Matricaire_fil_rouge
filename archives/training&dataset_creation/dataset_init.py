import argparse
import random
import shutil
from pathlib import Path

# =========================
# ARGUMENTS CLI
# =========================
parser = argparse.ArgumentParser(description="Split dataset YOLO (images + labels)")

parser.add_argument("--src", required=True, type=Path,
                    help="Dossier source contenant les images et fichiers .txt")
parser.add_argument("--out", required=True, type=Path,
                    help="Dossier de sortie (dataset YOLO)")

parser.add_argument("--train", type=float, default=0.7, help="Ratio train")
parser.add_argument("--val", type=float, default=0.2, help="Ratio validation")
parser.add_argument("--test", type=float, default=0.1, help="Ratio test")

parser.add_argument("--ext", default=".png", help="Extension des images (.png, .jpg)")
parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")

args = parser.parse_args()

# =========================
# VÉRIFICATIONS
# =========================
assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
    "Les ratios train/val/test doivent sommer à 1"

assert args.src.exists(), f"Dossier source introuvable : {args.src}"

random.seed(args.seed)

# =========================
# CRÉATION DES DOSSIERS
# =========================
for split in ["train", "val", "test"]:
    (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

# =========================
# LISTE DES IMAGES
# =========================
images = sorted(args.src.glob(f"*{args.ext}"))

if len(images) == 0:
    raise RuntimeError("Aucune image trouvée")

# Vérifier correspondance image ↔ label
for img in images:
    if not img.with_suffix(".txt").exists():
        raise FileNotFoundError(f"Annotation manquante pour {img.name}")

# =========================
# SHUFFLE + SPLIT
# =========================
random.shuffle(images)

n_total = len(images)
n_train = int(n_total * args.train)
n_val = int(n_total * args.val)

train_imgs = images[:n_train]
val_imgs = images[n_train:n_train + n_val]
test_imgs = images[n_train + n_val:]

# =========================
# COPIE
# =========================
def copy_pairs(img_list, split):
    for img in img_list:
        shutil.copy(img, args.out / "images" / split / img.name)
        shutil.copy(img.with_suffix(".txt"),
                    args.out / "labels" / split / img.with_suffix(".txt").name)

copy_pairs(train_imgs, "train")
copy_pairs(val_imgs, "val")
copy_pairs(test_imgs, "test")

# =========================
# RÉSUMÉ
# =========================
print("Split terminé avec succès")
print(f"Source : {args.src}")
print(f"Destination : {args.out}")
print(f"Train : {len(train_imgs)} images")
print(f"Val   : {len(val_imgs)} images")
print(f"Test  : {len(test_imgs)} images")