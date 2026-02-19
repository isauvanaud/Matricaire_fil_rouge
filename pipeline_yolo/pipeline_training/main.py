from logging import config
from pathlib import Path
import sys

ROOT = Path(__file__).parent 
sys.path.append(str(ROOT / "src"))

import os
from config import Config
from preprocessing_pipeline.database_preparation import set_databases
from preprocessing_pipeline.database_preparation import dataset_preparation
from training.run_function import generate_yolo_yaml, infer_dataset_type, iteration_yolo
import shutil
import torch

assert torch.cuda.is_available(), "CUDA non disponible"
print("GPU :", torch.cuda.get_device_name(0))
print("CUDA PyTorch :", torch.version.cuda)

def main():
    # Chargement configuration
    cfg = Config()

    # Dossier de destination relatif
    data_path = "./data"
    source_path = "./data/raw"
    database_path = "./data/intermed"
    runs_path = "./runs"

    # Parcours de tous les dossiers dans data_path
    for folder_name in os.listdir(source_path):
        print(folder_name)
        folder_path = os.path.join(source_path, folder_name)
        if os.path.isdir(folder_path):  # on ne prend que les dossiers
            print(f"✔ Traitement de : {folder_name}")

            # Appel de votre fonction sur ce dossier
            # set_databases(
            #     soil_list=cfg.soil,
            #     source_path=folder_path,          # chaque sous-dossier comme source
            #     destination_path=data_path,  # où mettre les variantes
            #     database_name=folder_name,        # nom du dataset = nom du dossier
            #     mu=cfg.mu,
            #     sigma=cfg.sigma,
            #     max_tries=cfg.max_tries,
            #     background_class_id = cfg.background_class_id
            # )
            set_databases(
                soil_list=cfg.soil,
                source_path=folder_path,
                destination_path=data_path,
                database_name=folder_name,
                config=cfg,
                seed=None
            )
    print("✔ Tous les dossiers traités avec succès !")

    for folder_name in os.listdir(database_path):
        if "database" not in folder_name.lower():
            continue

        folder_path = os.path.join(database_path, folder_name)

        if os.path.isdir(folder_path):
            print(f"✔ Traitement de : {folder_name}")
            dataset_preparation(folder_path, folder_name, cfg.dataset_number)

    # Création de la structure de dossiers pour les runs
    os.makedirs(runs_path, exist_ok=True)
    # os.makedirs(os.path.join(runs_path, "outs"), exist_ok=True)

    outs_path = Path(runs_path) / "outs"
    outs_path.mkdir(parents=True, exist_ok=True)

    for d in os.listdir(database_path):
        full_path = os.path.join(database_path, d)

        if not os.path.isdir(full_path):
            continue
        if not d.startswith("dataset") or "_split" not in d:
            continue

        db_part, split_part = d.rsplit("_split", 1)
        split_number = split_part
        database_name = db_part

        dataset_type = infer_dataset_type(d, cfg)
        class_names = cfg.names[dataset_type]
        nc = cfg.nc[dataset_type]

        for model in cfg.modeles:
            for size in cfg.size:
                path = os.path.join(
                    runs_path,
                    model,
                    size,
                    database_name,
                    f"split_{split_number}"
                )
                os.makedirs(path, exist_ok=True)

                destination = os.path.join(path, os.path.basename(full_path))
                shutil.copytree(full_path, destination, dirs_exist_ok=True)
                print(f"Copié : {full_path} → {destination}")

                yaml_path = os.path.join(path, "dataset.yaml")

                generate_yolo_yaml(
                    yaml_path,
                    destination,
                    nc,
                    class_names
                )
                name_mod = model+size
                iteration_yolo(cfg.iteration, name_mod, path, yaml_path, cfg, outs_path, d)



if __name__ == "__main__":
    main()
