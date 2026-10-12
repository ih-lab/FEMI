"""
labelme_json_to_mask.py

Convert Labelme annotation JSON files into single-channel PNG masks
(embryo = 255, background = 0) at the annotated image's original size.
train_segmenter.py and eval_segmenter.py resize masks with their images.
Needs only Pillow.

Usage:
    python labelme_json_to_mask.py path/to/one.json [more.json ...]
    python labelme_json_to_mask.py --input-dir path/to/jsons --output-dir path/to/masks
"""

import argparse
import json
import os

from PIL import Image, ImageDraw

SUPPORTED_LABELS_HELP = (
    "By default, every shape's points are filled in and unioned "
    "regardless of label. Pass --label to restrict to shapes with that "
    "exact label (e.g. --label embryo) if a file has multiple objects."
)


def polygon_mask_from_labelme(json_path, label=None):
    """Fill all (or `label`-matching) shapes into one 0/255 "L" mask at the
    JSON's imageWidth x imageHeight. Returns (mask, the JSON's imagePath)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        if label is not None and shape.get("label") != label:
            continue
        points = [(round(x), round(y)) for x, y in shape["points"]]
        if len(points) < 2:
            continue
        if shape.get("shape_type", "polygon") == "rectangle":
            (x1, y1), (x2, y2) = points
            draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255)
        else:
            # Other shape types are filled as a polygon of their points.
            draw.polygon(points, fill=255)

    return mask, data.get("imagePath")


def convert_one(json_path, output_dir, label=None, suffix="_mask"):
    mask, image_path = polygon_mask_from_labelme(json_path, label=label)
    stem = os.path.splitext(os.path.basename(image_path or json_path))[0]
    out_path = os.path.join(output_dir, f"{stem}{suffix}.png")
    os.makedirs(output_dir, exist_ok=True)
    mask.save(out_path)
    return out_path


def collect_json_paths(jsons, input_dir):
    paths = list(jsons)
    if input_dir:
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(".json"):
                paths.append(os.path.join(input_dir, f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert Labelme polygon JSON annotations into binary mask PNGs "
                    "(0/255, single channel), at the annotated image's original resolution.",
        epilog=SUPPORTED_LABELS_HELP,
    )
    parser.add_argument("jsons", nargs="*", help="Labelme .json files.")
    parser.add_argument("--input-dir", default=None,
                        help="Also convert every .json file in this directory.")
    parser.add_argument("--output-dir", default="masks",
                        help="Mask output directory (default: masks).")
    parser.add_argument("--label", default=None,
                        help="Only use shapes with this label (default: all shapes).")
    parser.add_argument("--suffix", default="_mask",
                        help="Output filename suffix (default: _mask, so foo.jpg -> foo_mask.png).")
    args = parser.parse_args()

    json_paths = collect_json_paths(args.jsons, args.input_dir)
    if not json_paths:
        parser.error("No JSON files provided. Pass paths directly and/or use --input-dir.")

    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"WARNING: {json_path} does not exist, skipping.")
            continue
        out_path = convert_one(json_path, args.output_dir, label=args.label, suffix=args.suffix)
        print(f"{json_path} -> {out_path}")


if __name__ == "__main__":
    main()
