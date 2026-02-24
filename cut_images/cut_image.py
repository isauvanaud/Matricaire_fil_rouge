#!/usr/bin/env python3

from PIL import Image
import os
import argparse

def split_image_into_squares(image_path, output_dir, tile_size):
    img = Image.open(image_path)
    width, height = img.size

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    idx = 0
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            tile.save(
                os.path.join(output_dir, f"{image_name}_{idx}.png")
            )
            idx += 1


def process_folder(input_dir, output_dir, tile_size):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            split_image_into_squares(
                os.path.join(input_dir, file),
                output_dir,
                tile_size
            )


def main():
    parser = argparse.ArgumentParser(
        description="Découpe des images en carrés"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Dossier contenant les images"
    )
    parser.add_argument(
        "--output", "-o",
        default="tiles",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=256,
        help="Taille des carrés (pixels)"
    )

    args = parser.parse_args()

    process_folder(args.input, args.output, args.size)


if __name__ == "__main__":
    main()
