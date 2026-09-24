"""
run_segmenter_inspect.py

Standalone CLI tool to run ONLY the embryo segmenter (PyTorch) on a
handful of raw images, for manual visual inspection. Unlike
`preprocessing_images.py` (which is a batch-processing script wired to a
whole directory), this tool is meant for interactively checking a few
problematic images and inspecting every intermediate artifact.

Does NOT require TensorFlow/transformers/pandas — only the dependencies
listed in `EmbryoSegmenter/requirements-segmenter.txt` (torch, numpy,
pillow, opencv). See that file / the README section on running this
script for setup instructions.

The segmentation logic below implements the "Phase 1" fix from
SEGMENTER_ACCURACY_ROADMAP.md (root causes A/B) instead of reproducing the
production bug: rather than unioning every contour whose *circularity*
clears a threshold, it picks a single best-candidate contour using
border-rejection + a plausible-size cap + centrality — no circularity
check at all. See `select_best_contour()` in `eval_segmenter.py` and
SEGMENTER_ACCURACY_ROADMAP.md Option C item 1 for the reasoning.

The low-level pipeline primitives this tool depends on (`load_model`,
`load_image`, `segment_image`, `calculate_circularity`,
`automatic_brightness_and_contrast`, `find_bounding_box`,
`select_best_contour`, `squarify_bbox`) live in `eval_segmenter.py`
(imported below) — they're shared with that script's `--mode pipeline`
benchmark, which runs these same primitives end-to-end across several
named pipeline configurations (including this tool's exact recipe) to
produce quantitative Dice/IoU/HD95/bbox-IoU/failure-rate comparisons.
This tool is still duplicated from (not imported from)
`preprocessing_images.py`'s circularity-union logic (see
`eval_segmenter.py`'s module docstring for why) — production still uses
the old circularity-union logic until that fix is ported over there too.

For each input image, this script saves 2 files to --output-dir:
  <stem>_4_bbox_overlay.png       the input image with the final crop box drawn in red
                                   (and the squarified box in green)
  <stem>_6_cropped_224_fixed.png  the final crop, squared up (padded to a square, centered
                                   on the original box, clamped to image bounds) BEFORE
                                   resizing to 224x224, so the resize is uniform and the
                                   embryo's true shape is preserved (see `squarify_bbox()`).

It also prints per-image diagnostics that mirror the "Phase 0"
instrumentation recommended in SEGMENTER_ACCURACY_ROADMAP.md:
  - whether the "give up, use full image" fallback triggered
  - fraction of the 224x224 frame covered by the final bounding box, and
    whether it exceeds 70% (`possible_background_false_positive`) — a
    proxy for the "dish rim / light vignette mistaken for the embryo"
    failure mode; a crop that large is most likely a faulty detection
    rather than a genuinely embryo-filling frame
  - whether the final bounding box touches the image border (a proxy for
    either model under-recall or a genuinely clipped source image)
  - bbox_aspect_ratio (width/height of the crop box) and whether it's far
    enough from 1.0 to cause visible stretching in the production-faithful
    (warp) crop, once force-resized to a 224x224 square
  - selected_contour_circularity: circularity of the chosen contour,
    reported purely for diagnostic/manual-review purposes — it is NOT
    used to select or reject the contour

Usage:
    python run_segmenter_inspect.py path/to/image1.png path/to/image2.jpg
    python run_segmenter_inspect.py --input-dir path/to/raw_images --output-dir out/
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from eval_segmenter import (
    DEFAULT_WEIGHTS,
    automatic_brightness_and_contrast,
    calculate_circularity,
    find_bounding_box,
    load_image,
    load_model,
    segment_image,
    select_best_contour,
    squarify_bbox,
)
from train_segmenter import IMAGE_EXTS


def to_bgr_uint8(img):
    """Best-effort conversion of a loaded image array (grayscale, RGB, or
    RGBA) to a 3-channel BGR uint8 array, for drawing an OpenCV overlay."""
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

    # --- Fixed brightness (_2): use the automatic brightness correction function to improve the image quality.
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
    box_w = max(1, max_x - min_x)
    box_h = max(1, max_y - min_y)
    diagnostics["crop_area_fraction"] = round((box_w * box_h) / float(w * h), 3)
    diagnostics["possible_background_false_positive"] = bool(
        diagnostics["crop_area_fraction"] > background_area_flag_threshold)
    diagnostics["touches_border"] = bool(min_x <= 1 or min_y <= 1 or max_x >= w - 1 or max_y >= h - 1)
    diagnostics["bounding_box_xyxy"] = (int(min_x), int(min_y), int(max_x), int(max_y))

    bbox_aspect_ratio = round(box_w / box_h, 3)
    diagnostics["bbox_aspect_ratio"] = bbox_aspect_ratio
    # Ratios far from 1.0 mean the production resize-to-square step will
    # visibly stretch/warp the embryo (see squarify_bbox docstring / root
    # cause F). >20% asymmetry is a reasonable "visibly distorted" cutoff.
    diagnostics["likely_visible_distortion_in_prod_output"] = bool(
        bbox_aspect_ratio > 1.2 or bbox_aspect_ratio < (1 / 1.2))

    # --- Fixed crop (_6): square up the box first so the resize is
    # --- distortion-free. See squarify_bbox().
    sq_min_x, sq_min_y, sq_max_x, sq_max_y = squarify_bbox(min_x, min_y, max_x, max_y, w, h)
    fixed_cropped_image = img[sq_min_y:sq_max_y, sq_min_x:sq_max_x]
    fixed_cropped_image = Image.fromarray(np.clip(fixed_cropped_image, 0, 255).astype(np.uint8)).resize((224, 224))

    os.makedirs(output_dir, exist_ok=True)

    overlay = to_bgr_uint8(img)
    cv2.rectangle(overlay, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)
    cv2.rectangle(overlay, (sq_min_x, sq_min_y), (sq_max_x, sq_max_y), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, f"{stem}_4_bbox_overlay.png"), overlay)

    fixed_cropped_image.save(os.path.join(output_dir, f"{stem}_6_cropped_224_fixed.png"))

    return diagnostics


def collect_image_paths(images, input_dir):
    paths = list(images)
    if input_dir:
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(input_dir, f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Run only the embryo segmenter (no TensorFlow/FEMI needed) "
                    "on a handful of raw images for manual inspection.")
    parser.add_argument("images", nargs="*", help="One or more image file paths to process.")
    parser.add_argument("--input-dir", default=None,
                        help="Alternatively (or additionally), process every supported image in this directory.")
    parser.add_argument("--output-dir", default="segmenter_inspection_output",
                        help="Directory to write diagnostic outputs to (default: ./segmenter_inspection_output).")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to the segmenter .pt weights file.")
    parser.add_argument("--max-area-frac", type=float, default=0.9,
                        help="Reject a candidate contour as an implausible dish-rim/vignette "
                             "detection if its area exceeds this fraction of the full frame "
                             "(default: 0.9). See select_best_contour().")
    parser.add_argument("--border-margin", type=int, default=2,
                        help="Pixel margin used to decide whether a candidate contour "
                             "touches the image border (default: 2). See select_best_contour().")
    parser.add_argument("--min-area-frac", type=float, default=0.1,
                        help="Discard candidate contours smaller than this fraction of the "
                             "frame BEFORE applying the border/size preference logic (default: "
                             "0.1, i.e. 10%%). Without this, a tiny noise speck that happens to "
                             "not touch the border can beat a much larger, genuine detection "
                             "that grazes the border by a couple pixels - see "
                             "select_best_contour() and SEGMENTER_ACCURACY_ROADMAP.md Root "
                             "cause B.")
    parser.add_argument("--background-area-flag-threshold", type=float, default=0.7,
                        help="Flag an image's 'possible_background_false_positive' diagnostic "
                             "if the final crop covers more than this fraction of the frame "
                             "(default: 0.7) — most likely a faulty detection rather than a "
                             "genuinely embryo-filling frame.")
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
            if k != "image":
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
        print(f"  - Likely visible aspect-ratio distortion in the warp-crop output: {n_distorted} "
              f"(compare against the _6 squarified output)")


if __name__ == "__main__":
    main()
