"""
run_segmenter_inspect.py

Run only the embryo segmenter on a few images and save every intermediate
step for visual inspection. Uses the config C pipeline (brightness first,
select_best_contour, squarify crop) with primitives from eval_segmenter.py.
Needs only torch, numpy, pillow, and opencv.

Per image, saved to --output-dir:
  <stem>_2_brightness_adjusted.png  what the model sees
  <stem>_3_raw_mask.png             thresholded mask before contour selection
  <stem>_4_bbox_overlay.png         final box (red) and squared box (green)
  <stem>_6_cropped_224_fixed.png    the squared 224x224 crop

Printed diagnostics: full-image fallback, crop area fraction (flagged as a
possible dish/vignette false positive above the threshold), border touch,
bbox aspect ratio (flagged if a direct resize would visibly distort), and
the chosen contour's circularity (informational only).

Images that trip a --review-on criterion get their outputs copied to
<output-dir>/needs_review/ along with needs_review.csv.

Usage:
    python run_segmenter_inspect.py path/to/image1.png path/to/image2.jpg
    python run_segmenter_inspect.py --input-dir path/to/raw_images --output-dir out/
"""

import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

from eval_segmenter import (
    DEFAULT_SKIP_WEIGHTS,
    automatic_brightness_and_contrast,
    calculate_circularity,
    compute_crop_diagnostics,
    find_bounding_box,
    load_image,
    load_model,
    segment_image,
    select_best_contour,
    squarify_bbox,
)
from train_segmenter import IMAGE_EXTS

# --review-on choice -> the diagnostics key that triggers it.
REVIEW_CRITERIA = {
    "fallback": "fallback_full_image_triggered",
    "background_fp": "possible_background_false_positive",
    "border": "touches_border",
    "distortion": "likely_visible_distortion_in_prod_output",
}


def to_bgr_uint8(img):
    """Grayscale/RGB/RGBA array -> BGR uint8, for OpenCV drawing."""
    arr = np.clip(img, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)


def process_one_image(model, img_path, output_dir, max_area_frac=0.9, border_margin=2,
                       background_area_flag_threshold=0.7, min_area_frac=0.1):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    diagnostics = {
        "image": img_path,
        "fallback_full_image_triggered": False,
    }

    img = load_image(img_path)
    if np.max(img) == 0:
        diagnostics["note"] = "Blank image (max pixel value is 0) - skipped, matching production behavior."
        return diagnostics

    bright_img = automatic_brightness_and_contrast(img)
    raw_mask = segment_image(model, bright_img)
    final_mask, selected_contour = select_best_contour(raw_mask, max_area_frac, border_margin, min_area_frac)

    if np.max(final_mask) == 0:
        diagnostics["fallback_full_image_triggered"] = True
        final_mask = np.ones_like(final_mask)
        selected_contour = None

    diagnostics["selected_contour_circularity"] = (
        round(float(calculate_circularity(selected_contour)), 3) if selected_contour is not None else None
    )

    min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
    h, w = img.shape[0], img.shape[1]
    # "warp" so the distortion flag reports what an unsquared resize would do.
    crop_diag = compute_crop_diagnostics((min_x, min_y, max_x, max_y), w, h, "warp",
                                         background_area_flag_threshold)
    diagnostics["crop_area_fraction"] = round(crop_diag["crop_area_fraction"], 3)
    diagnostics["possible_background_false_positive"] = crop_diag["possible_background_false_positive"]
    diagnostics["touches_border"] = crop_diag["touches_border"]
    diagnostics["bounding_box_xyxy"] = (int(min_x), int(min_y), int(max_x), int(max_y))
    diagnostics["bbox_aspect_ratio"] = round(crop_diag["bbox_aspect_ratio"], 3)
    diagnostics["likely_visible_distortion_in_prod_output"] = crop_diag["likely_visible_distortion"]

    sq_min_x, sq_min_y, sq_max_x, sq_max_y = squarify_bbox(min_x, min_y, max_x, max_y, w, h)
    fixed_cropped_image = img[sq_min_y:sq_max_y, sq_min_x:sq_max_x]
    fixed_cropped_image = Image.fromarray(np.clip(fixed_cropped_image, 0, 255).astype(np.uint8)).resize((224, 224))

    os.makedirs(output_dir, exist_ok=True)

    bright_path = os.path.join(output_dir, f"{stem}_2_brightness_adjusted.png")
    cv2.imwrite(bright_path, to_bgr_uint8(bright_img))

    raw_mask_path = os.path.join(output_dir, f"{stem}_3_raw_mask.png")
    cv2.imwrite(raw_mask_path, (raw_mask * 255).astype(np.uint8))

    overlay = to_bgr_uint8(img)
    cv2.rectangle(overlay, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)
    cv2.rectangle(overlay, (sq_min_x, sq_min_y), (sq_max_x, sq_max_y), (0, 255, 0), 1)
    overlay_path = os.path.join(output_dir, f"{stem}_4_bbox_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    cropped_path = os.path.join(output_dir, f"{stem}_6_cropped_224_fixed.png")
    fixed_cropped_image.save(cropped_path)

    diagnostics["output_files"] = [bright_path, raw_mask_path, overlay_path, cropped_path]
    return diagnostics


def copy_flagged_for_review(all_diagnostics, review_dir, review_on):
    """Copy flagged images' outputs to `review_dir` and write needs_review.csv."""
    flagged = []
    for d in all_diagnostics:
        reasons = [name for name in review_on if d.get(REVIEW_CRITERIA[name])]
        if reasons:
            flagged.append((d, reasons))
    if not flagged:
        return []

    os.makedirs(review_dir, exist_ok=True)
    rows = []
    for d, reasons in flagged:
        for f in d.get("output_files", []):
            shutil.copy2(f, review_dir)
        rows.append({
            "image": d["image"],
            "reasons": ";".join(reasons),
            "crop_area_fraction": d.get("crop_area_fraction"),
            "bbox_aspect_ratio": d.get("bbox_aspect_ratio"),
            "selected_contour_circularity": d.get("selected_contour_circularity"),
            "bounding_box_xyxy": d.get("bounding_box_xyxy"),
        })
    with open(os.path.join(review_dir, "needs_review.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def collect_image_paths(images, input_dir):
    paths = list(images)
    if input_dir:
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(input_dir, f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Run only the embryo segmenter on a few images for manual inspection.")
    parser.add_argument("images", nargs="*", help="Image file paths.")
    parser.add_argument("--input-dir", default=None,
                        help="Also process every image in this directory.")
    parser.add_argument("--output-dir", default="segmenter_inspection_output",
                        help="Output directory (default: segmenter_inspection_output).")
    parser.add_argument("--weights", default=DEFAULT_SKIP_WEIGHTS,
                        help="Segmenter checkpoint (default: weights_skip/latest.pt).")
    parser.add_argument("--max-area-frac", type=float, default=0.9,
                        help="Max contour area as a frame fraction (default: 0.9).")
    parser.add_argument("--border-margin", type=int, default=2,
                        help="Border-touch margin in px (default: 2).")
    parser.add_argument("--min-area-frac", type=float, default=0.1,
                        help="Min contour area as a frame fraction, to drop noise specks (default: 0.1).")
    parser.add_argument("--background-area-flag-threshold", type=float, default=0.7,
                        help="Flag crops covering more than this frame fraction (default: 0.7).")
    parser.add_argument("--review-on", nargs="+", choices=sorted(REVIEW_CRITERIA),
                        default=["fallback"],
                        help="Flags that send an image to <output-dir>/needs_review/ (default: fallback).")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.images, args.input_dir)
    if not image_paths:
        parser.error("No images provided. Pass image paths directly and/or use --input-dir.")

    if not os.path.exists(args.weights):
        print(f"ERROR: weights file not found at {args.weights}", file=sys.stderr)
        print("Pass --weights /path/to/your/weights.pt if it lives elsewhere.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading segmenter weights from {args.weights} ...")
    model = load_model(args.weights)
    print(f"Model loaded ({type(model).__name__}).\n")

    all_diagnostics = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"WARNING: {img_path} does not exist, skipping.")
            continue
        print(f"Processing {img_path} ...")
        diag = process_one_image(model, img_path, args.output_dir,
                                  max_area_frac=args.max_area_frac,
                                  border_margin=args.border_margin,
                                  background_area_flag_threshold=args.background_area_flag_threshold,
                                  min_area_frac=args.min_area_frac)
        all_diagnostics.append(diag)
        for k, v in diag.items():
            if k not in ("image", "output_files"):
                print(f"    {k}: {v}")
        print()

    print(f"Done. Per-image outputs saved to: {os.path.abspath(args.output_dir)}")

    n = len(all_diagnostics)
    if n:
        n_bg_flag = sum(1 for d in all_diagnostics if d.get("possible_background_false_positive"))
        n_border_flag = sum(1 for d in all_diagnostics if d.get("touches_border"))
        n_full_fallback = sum(1 for d in all_diagnostics if d.get("fallback_full_image_triggered"))
        n_distorted = sum(1 for d in all_diagnostics if d.get("likely_visible_distortion_in_prod_output"))
        print(f"\nSummary across {n} image(s) (see SEGMENTER_ACCURACY_ROADMAP.md for context):")
        print(f"  - Possible dish/vignette false-positive (crop area > "
              f"{args.background_area_flag_threshold * 100:.0f}% of frame, likely faulty detection): {n_bg_flag}")
        print(f"  - Bounding box touches image border (possible under-recall or clipped source): {n_border_flag}")
        print(f"  - Full-image fallback triggered (no valid region found at all): {n_full_fallback}")
        print(f"  - Likely visible aspect-ratio distortion in the warp-crop output (could be abnormal development): {n_distorted} "
              f"(compare against the _6 squarified output)")

        review_dir = os.path.join(args.output_dir, "needs_review")
        flagged = copy_flagged_for_review(all_diagnostics, review_dir, args.review_on)
        print(f"\nFlagged for manual review (--review-on {' '.join(args.review_on)}): {len(flagged)}")
        if flagged:
            print(f"  Copied their outputs + needs_review.csv to: {os.path.abspath(review_dir)}")


if __name__ == "__main__":
    main()
