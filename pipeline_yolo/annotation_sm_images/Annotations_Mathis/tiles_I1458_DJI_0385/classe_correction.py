import os

def force_class_to_zero(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)

            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # YOLO attendu : class x y w h (ou plus de champs)
                parts[0] = "0"
                new_lines.append(" ".join(parts))

            with open(path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

    print("Tous les fichiers YOLO ont été mis à jour (classe → 0).")

if __name__ == "__main__":
    force_class_to_zero()
