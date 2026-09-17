"""
labelme_json_to_mask.py

Converts Labelme polygon annotation JSON files into binary mask PNGs in
the same convention used elsewhere in this repo (see e.g.
`select_best_contour` in run_segmenter_inspect.py): single-channel
uint8 PNG, embryo region = 255, background = 0, same width/height as
the original annotated image (Labelme's `imageWidth`/`imageHeight`
fields) - NOT resized to the model's 224x224 input size. Resize the
mask together with its image (using nearest-neighbor, to keep it
binary) at training/eval time instead of baking a resize in here, so
mask and image always end up pixel-aligned regardless of what input
size a given model run uses.

Depends only on Pillow (already a dependency elsewhere in this repo),
no OpenCV needed. Runs entirely locally - reads/writes only local
files, no network calls - safe for sensitive images.

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
    """Rasterizes all (or label-matching) polygon/shape point lists in a
    Labelme JSON file into a single binary mask, at the original
    imageWidth x imageHeight resolution recorded in the JSON.

    Returns (mask, image_path) where mask is a "L"-mode (uint8) PIL
    Image with values in {0, 255}, and image_path is the JSON's
    recorded `imagePath` (for naming the output file after the source
    image rather than the JSON).
    """
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
            # polygon, circle, line, etc. - filling the raw point list as a
            # polygon covers Labelme's common shape types well enough here,
            # since annotation is always a single closed embryo outline.
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
    parser.add_argument("jsons", nargs="*", help="One or more Labelme .json annotation files.")
    parser.add_argument("--input-dir", default=None,
                        help="Alternatively (or additionally), convert every .json file in this directory.")
    parser.add_argument("--output-dir", default="masks",
                        help="Directory to write mask PNGs to (default: ./masks).")
    parser.add_argument("--label", default=None,
                        help="Only rasterize shapes with this exact label (default: all shapes, unioned).")
    parser.add_argument("--suffix", default="_mask",
                        help="Suffix appended to the source image's stem for the output filename "
                             "(default: _mask, e.g. foo.jpg -> foo_mask.png).")
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
