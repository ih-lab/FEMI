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

The segmentation + contour-filtering + fallback logic below intentionally
mirrors `process_image()` in `preprocessing_images.py` byte-for-byte in
behavior (including the known "union of all circularity-passing contours"
issue described in SEGMENTER_ACCURACY_ROADMAP.md), so that what you see
here matches what production actually does today. It is duplicated rather
than imported because `preprocessing_images.py` runs directory-listing and
model-loading side effects at import time (it's a script, not a library).

For each input image, this script saves 6 files to --output-dir:
  <stem>_1_input.png              the 224x224 resized image actually fed to the model
  <stem>_2_raw_mask.png           the raw thresholded model mask (before contour filtering)
  <stem>_3_filtered_mask.png      the mask after circularity filtering (matches production)
  <stem>_4_bbox_overlay.png       the input image with the final crop box drawn in red
  <stem>_5_cropped_224.png        the final cropped + resized output EXACTLY as
                                   preprocessing_images.py produces it today — including
                                   its non-aspect-ratio-preserving resize (see root cause F
                                   in SEGMENTER_ACCURACY_ROADMAP.md). If the bounding box
                                   isn't square, THIS OUTPUT WILL BE VISIBLY STRETCHED/WARPED.
                                   That is a genuine production bug, not an artifact of this
                                   inspection tool - see `bbox_aspect_ratio` in the printed
                                   diagnostics below.
  <stem>_6_cropped_224_fixed.png  the SAME crop, but squared up (padded to a square, centered
                                   on the original box, clamped to image bounds) BEFORE
                                   resizing to 224x224, so the resize is uniform and the
                                   embryo's true shape is preserved. Compare this against
                                   _5 to see the distortion fix in action.

It also prints per-image diagnostics that mirror the "Phase 0"
instrumentation recommended in SEGMENTER_ACCURACY_ROADMAP.md:
  - whether the brightness-correction fallback triggered
  - whether the "give up, use full image" fallback triggered
  - fraction of the 224x224 frame covered by the final bounding box
    (a high fraction, e.g. >85%, is a proxy for the "dish rim / light
    vignette mistaken for the embryo" failure mode)
  - whether the final bounding box touches the image border (a proxy for
    either model under-recall or a genuinely clipped source image)
  - bbox_aspect_ratio (width/height of the crop box) and whether it's far
    enough from 1.0 to cause visible stretching in the production-faithful
    _5 output once force-resized to a 224x224 square

Usage:
    python run_segmenter_inspect.py path/to/image1.png path/to/image2.jpg
    python run_segmenter_inspect.py --input-dir path/to/raw_images --output-dir out/
"""

import argparse
import os
import sys
import warnings

# automatic_brightness_and_contrast() below (copied verbatim from
# preprocessing_images.py for behavioral fidelity) triggers a harmless
# NumPy scalar-conversion DeprecationWarning; silenced here for cleaner
# console output. This does not affect any computed values.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import torch
from PIL import Image

from EmbryoSegmenter.SegmenterModel import SegmenterModel

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "EmbryoSegmenter", "weights", "epoch_102_loss_0.098710.pt",
)
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_image(image_path, size=(224, 224)):
    """Same behavior as `load_image` in preprocessing_images.py: open,
    resize to a fixed size, and convert to a numpy array (whatever mode
    the source image is in — grayscale or RGB)."""
    img = Image.open(image_path)
    img = img.resize(size)
    return np.array(img)


def segment_image(model, frame):
    """Same behavior as `segment_image` in preprocessing_images.py: one
    forward pass through SegmenterModel, thresholded at 0.5. Returns just
    the binary mask (this tool doesn't need the discarded
    `segmented_images` pixel-masked output — see SEGMENTER_ACCURACY_ROADMAP.md
    §1 for why that output isn't actually used downstream in production
    either)."""
    frame = frame / 255.0
    if len(frame.shape) > 2:
        frame = frame[:, :, 0]
    select_frames = frame[None, None, ...]
    select_frames = np.array(select_frames)
    batch = torch.tensor(select_frames).float()
    with torch.no_grad():
        outputs = model(batch)
    mask = outputs.detach().numpy()[0, 0, :, :].copy()
    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1
    return mask


def calculate_circularity(contour):
    """Identical to preprocessing_images.py."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter ** 2))


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    """Identical to preprocessing_images.py."""
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    if (maximum_gray - minimum_gray) == 0:
        return image
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def find_bounding_box(mask):
    """Identical to preprocessing_images.py."""
    if np.sum(mask) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    return (min_x, min_y, max_x, max_y)


def filter_mask_by_circularity(mask, min_circularity):
    """Reproduces the CURRENT production contour-filtering logic exactly:
    finds external contours, keeps every one whose circularity clears the
    threshold, and draws ALL of them (union, not just the single best
    candidate) onto the final mask. See SEGMENTER_ACCURACY_ROADMAP.md
    root causes A/B for why this union behavior is a likely source of the
    "dish rim treated as embryo" and "irregular embryo excluded" failure
    modes — this tool deliberately keeps that behavior so what you observe
    here matches production, rather than a hypothetical fixed version."""
    cv2mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
    final_mask = np.zeros_like(cv2mask)
    cv2.drawContours(final_mask, circular_contours, -1, (255,), thickness=cv2.FILLED)
    return (final_mask / 255.0).astype(float)


def squarify_bbox(min_x, min_y, max_x, max_y, img_w, img_h):
    """
    Expand a (possibly non-square) bounding box into a square box of side
    `max(width, height)`, centered on the original box's center, then
    shift it (without resizing) so it stays fully within the image
    bounds. If the ideal square side is larger than the smaller image
    dimension, it's capped to fit (can't make a square bigger than the
    image without padding).

    This exists to fix a real distortion bug in the current production
    pipeline: `preprocessing_images.py` crops directly to the (often
    non-square) mask bounding box and then does
    `Image.resize((224, 224))`, which does NOT preserve aspect ratio and
    visibly warps the embryo whenever the box isn't square (see root
    cause F in SEGMENTER_ACCURACY_ROADMAP.md). Cropping a SQUARE region
    first means the final resize to 224x224 is a uniform scale with no
    distortion.
    """
    box_w = max_x - min_x
    box_h = max_y - min_y
    side = max(box_w, box_h)
    side = min(side, img_w, img_h)  # can't exceed image bounds without padding

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    new_min_x = int(round(cx - side / 2.0))
    new_min_y = int(round(cy - side / 2.0))

    # Shift (don't shrink) the square so it lies fully within the image.
    new_min_x = max(0, min(new_min_x, img_w - side))
    new_min_y = max(0, min(new_min_y, img_h - side))

    new_max_x = new_min_x + side
    new_max_y = new_min_y + side
    return int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)


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


def process_one_image(model, img_path, output_dir, min_circularity):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    diagnostics = {
        "image": img_path,
        "fallback_brightness_triggered": False,
        "fallback_full_image_triggered": False,
    }

    img = load_image(img_path)
    if np.max(img) == 0:
        diagnostics["note"] = "Blank image (max pixel value is 0) - skipped, matching production behavior."
        return diagnostics

    raw_mask = segment_image(model, img)
    final_mask = filter_mask_by_circularity(raw_mask, min_circularity)

    if np.max(final_mask) == 0:
        diagnostics["fallback_brightness_triggered"] = True
        bright_img = automatic_brightness_and_contrast(img)
        raw_mask = segment_image(model, bright_img)
        final_mask = filter_mask_by_circularity(raw_mask, min_circularity)
        if np.sum(final_mask) == 0:
            diagnostics["fallback_full_image_triggered"] = True
            final_mask = np.ones_like(final_mask)

    min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
    h, w = img.shape[0], img.shape[1]
    box_w = max(1, max_x - min_x)
    box_h = max(1, max_y - min_y)
    diagnostics["crop_area_fraction"] = round((box_w * box_h) / float(w * h), 3)
    diagnostics["touches_border"] = bool(min_x <= 1 or min_y <= 1 or max_x >= w - 1 or max_y >= h - 1)
    diagnostics["bounding_box_xyxy"] = (int(min_x), int(min_y), int(max_x), int(max_y))

    bbox_aspect_ratio = round(box_w / box_h, 3)
    diagnostics["bbox_aspect_ratio"] = bbox_aspect_ratio
    # Ratios far from 1.0 mean the production resize-to-square step will
    # visibly stretch/warp the embryo (see squarify_bbox docstring / root
    # cause F). >20% asymmetry is a reasonable "visibly distorted" cutoff.
    diagnostics["likely_visible_distortion_in_prod_output"] = bool(
        bbox_aspect_ratio > 1.2 or bbox_aspect_ratio < (1 / 1.2))

    # --- Production-faithful crop (_5): reproduces the existing
    # --- non-aspect-ratio-preserving resize bug exactly.
    cropped_image = img[min_y:max_y, min_x:max_x]
    cropped_image = Image.fromarray(np.clip(cropped_image, 0, 255).astype(np.uint8)).resize((224, 224))

    # --- Fixed crop (_6): square up the box first so the resize is
    # --- distortion-free. See squarify_bbox().
    sq_min_x, sq_min_y, sq_max_x, sq_max_y = squarify_bbox(min_x, min_y, max_x, max_y, w, h)
    fixed_cropped_image = img[sq_min_y:sq_max_y, sq_min_x:sq_max_x]
    fixed_cropped_image = Image.fromarray(np.clip(fixed_cropped_image, 0, 255).astype(np.uint8)).resize((224, 224))

    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{stem}_1_input.png"))
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{stem}_2_raw_mask.png"))
    Image.fromarray((final_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{stem}_3_filtered_mask.png"))

    overlay = to_bgr_uint8(img)
    cv2.rectangle(overlay, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)
    cv2.rectangle(overlay, (sq_min_x, sq_min_y), (sq_max_x, sq_max_y), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, f"{stem}_4_bbox_overlay.png"), overlay)

    cropped_image.save(os.path.join(output_dir, f"{stem}_5_cropped_224.png"))
    fixed_cropped_image.save(os.path.join(output_dir, f"{stem}_6_cropped_224_fixed.png"))

    return diagnostics


def collect_image_paths(images, input_dir):
    paths = list(images)
    if input_dir:
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(SUPPORTED_EXTS):
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
    parser.add_argument("--min-circularity", type=float, default=0.7,
                        help="Circularity threshold, matches the production default of 0.7.")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.images, args.input_dir)
    if not image_paths:
        parser.error("No images provided. Pass image paths directly and/or use --input-dir.")

    if not os.path.exists(args.weights):
        print(f"ERROR: weights file not found at {args.weights}", file=sys.stderr)
        print("Pass --weights /path/to/your/weights.pt if it lives elsewhere.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading segmenter weights from {args.weights} ...")
    model = SegmenterModel()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    print("Model loaded.\n")

    all_diagnostics = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"WARNING: {img_path} does not exist, skipping.")
            continue
        print(f"Processing {img_path} ...")
        diag = process_one_image(model, img_path, args.output_dir, args.min_circularity)
        all_diagnostics.append(diag)
        for k, v in diag.items():
            if k != "image":
                print(f"    {k}: {v}")
        print()

    print(f"Done. Per-image outputs saved to: {os.path.abspath(args.output_dir)}")

    n = len(all_diagnostics)
    if n:
        n_bg_flag = sum(1 for d in all_diagnostics if d.get("crop_area_fraction", 0) > 0.85)
        n_border_flag = sum(1 for d in all_diagnostics if d.get("touches_border"))
        n_fallback = sum(1 for d in all_diagnostics if d.get("fallback_brightness_triggered"))
        n_full_fallback = sum(1 for d in all_diagnostics if d.get("fallback_full_image_triggered"))
        n_distorted = sum(1 for d in all_diagnostics if d.get("likely_visible_distortion_in_prod_output"))
        print(f"\nSummary across {n} image(s) (see SEGMENTER_ACCURACY_ROADMAP.md for context):")
        print(f"  - Possible dish/vignette false-positive (crop area > 85% of frame): {n_bg_flag}")
        print(f"  - Bounding box touches image border (possible under-recall or clipped source): {n_border_flag}")
        print(f"  - Brightness-correction fallback triggered: {n_fallback}")
        print(f"  - Full-image fallback triggered (no valid region found at all): {n_full_fallback}")
        print(f"  - Likely visible aspect-ratio distortion in the _5 (production-faithful) crop: {n_distorted} "
              f"(compare _5 vs _6 outputs for these)")


if __name__ == "__main__":
    main()
