import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import os
import random

def infer_dataset_type(folder_name, config):
    """
    Trouve la clé de dataset correspondant au nom du dossier.
    """
    for key in config.names.keys():
        if key in folder_name:
            return key
    raise ValueError(f"Aucun type de dataset reconnu dans : {folder_name}")


def generate_yolo_yaml(output_path, dataset_path, nc, class_names):
    """
    Génère un fichier YAML pour YOLO.
    
    Parameters:
        output_path (str or Path): chemin du fichier YAML à créer
        dataset_path (str or Path): chemin racine des images
        nc (int): nombre de classes
        class_names (list): liste des noms de classes
    """
    # structure du YAML
    data_dict = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": class_names
    }

    # écriture YAML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(data_dict, f, sort_keys=False)

    print(f"YAML généré : {output_path}")


def yolo_run(it, name_mod, dataset_path,yaml_path, config, path_outs, name_dataset):

    model = YOLO(name_mod+".pt")
    dataset_path = Path(dataset_path).resolve()
    yaml_dir = Path(yaml_path).resolve()
    print(yaml_dir )
    random_seed = random.randint(0, 1_000_000)

    # ======================
    # TRAIN
    # ======================
    results = model.train(
        data=str(yaml_dir),
        imgsz=config.imgsz,
        epochs=config.epochs,
        batch=config.batch,
        patience=config.patience,
        workers=8,
        cache=True,
        device=0,
        project=str(dataset_path),
        name="runs",
        exist_ok=True,
        seed=random_seed
    )

    # ======================
    # CHEMIN DU BEST MODEL
    # ======================
    best_model_path = Path(dataset_path, "runs", "weights", "best.pt")

    # ======================
    # VALIDATION DU BEST
    # ======================
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=yaml_dir)

    # ======================
    # EXTRACTION MÉTRIQUES
    # ======================
    metrics_dict = {
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map
    }

    # ======================
    # ÉCRITURE TXT
    # ======================
    Path(path_outs).mkdir(parents=True, exist_ok=True)

    txt_name = f"{name_mod}_{name_dataset}_iteration_{it}.txt"
    txt_path = Path(path_outs, txt_name)

    with open(txt_path, "w") as f:
        f.write(f"Model: {name_mod}\n")
        f.write(f"Dataset: {name_dataset}\n")
        f.write(f"Iteration: {it}\n\n")

        for k, v in metrics_dict.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"Métriques sauvegardées dans : {txt_path}")


def iteration_yolo(iterations, name_mod, dataset_path,yaml_path, config, path_outs, name_dataset):
    for it in range(iterations):
        print(f"=== Itération {it+1}/{iterations} ===")
        yolo_run(it, name_mod, dataset_path, yaml_path, config, path_outs, name_dataset)