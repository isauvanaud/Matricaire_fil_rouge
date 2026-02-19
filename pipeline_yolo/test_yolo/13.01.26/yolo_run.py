from ultralytics import YOLO

# Chargement du modèle
model = YOLO("best.pt")

# Prédiction sur toutes les images du dossier "tile"
model.predict(
    source="tile",
    save=True,
    show_labels=False,
    show_conf=False
)