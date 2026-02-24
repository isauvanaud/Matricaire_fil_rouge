import re
from pathlib import Path
import json
import pandas as pd


def load_flower_results(folder, target_class="flower"):
    """
    Charge les JSON YOLO et conserve uniquement la classe cible.
    Extrait toutes les métriques disponibles.
    """
    records = []

    for file in Path(folder).glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        model = data["model"]
        dataset = data["dataset"]
        iteration = data["iteration"]

        for cls in data["summary"]:
            # votre JSON utilise "Class"
            if cls.get("Class", "").lower() != target_class.lower():
                continue

            row = {
                "model": model,
                "dataset": dataset,
                "iteration": iteration,

                # métriques détection
                "precision": cls.get("Box-P"),
                "recall": cls.get("Box-R"),
                "f1": cls.get("Box-F1"),

                # métriques COCO
                "map50": cls.get("mAP50"),
                "map50_95": cls.get("mAP50-95"),

                # infos dataset
                "instances": cls.get("Instances"),
                "images": cls.get("Images"),
            }

            records.append(row)

    if not records:
        raise ValueError("Aucune entrée trouvée pour la classe cible.")

    df = pd.DataFrame(records)

    # extraction dataset parent & split
    df["dataset_name"] = df["dataset"].str.replace(r"_split\d+", "", regex=True)
    df["split"] = df["dataset"].str.extract(r"_split(\d+)").astype(int)

    return df


def compute_iteration_stats(df):
    """
    Variabilité entre itérations d’un même split.
    """
    return (
        df.groupby(["model", "dataset"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            map50_mean=("map50", "mean"),
            map50_95_mean=("map50_95", "mean"),
            map50_95_std=("map50_95", "std"),
            iterations=("iteration", "count")
        )
        .reset_index()
    )


def compute_dataset_stats(df):
    """
    Performance globale par dataset (tous splits + itérations).
    """
    return (
        df.groupby(["model", "dataset_name"])
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            map50_mean=("map50", "mean"),
            map50_95_mean=("map50_95", "mean"),
            map50_95_std=("map50_95", "std"),
            instances_total=("instances", "sum"),
            images_total=("images", "sum"),
            n_runs=("iteration", "count")
        )
        .reset_index()
    )


def compare_models(df):
    """
    Comparaison globale des modèles (priorité mAP50-95).
    """
    return (
        df.groupby("model")
        .agg(
            map50_95_mean=("map50_95", "mean"),
            map50_95_std=("map50_95", "std"),
            map50_mean=("map50", "mean"),
            f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean")
        )
        .sort_values("map50_95_mean", ascending=False)
        .reset_index()
    )


def generate_flower_statistics(json_folder, out_folder="stats"):
    """
    Pipeline complet d'analyse pour la classe flower.
    """
    out = Path(out_folder)
    out.mkdir(exist_ok=True)

    df = load_flower_results(json_folder)

    iter_stats = compute_iteration_stats(df)
    dataset_stats = compute_dataset_stats(df)
    model_stats = compare_models(df)

    df.to_csv(out / "flower_raw_results.csv", index=False)
    iter_stats.to_csv(out / "flower_iteration_stats.csv", index=False)
    dataset_stats.to_csv(out / "flower_dataset_stats.csv", index=False)
    model_stats.to_csv(out / "flower_model_comparison.csv", index=False)

    print("✔ Stats générées dans :", out.resolve())

    # return iter_stats, dataset_stats, model_stats, df


# # ======================
# # EXECUTION
# # ======================

# iter_stats, dataset_stats, model_stats, raw_df = generate_flower_statistics(
#     json_folder="./runs/outs",
#     out_folder="./runs/stats"
# )

# print("\n=== Comparaison modèles ===")
# print(model_stats)
