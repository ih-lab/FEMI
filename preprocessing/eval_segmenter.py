"""
eval_segmenter.py

Benchmarks the embryo segmenter (`SegmenterModel` and `SegmenterModelSkip`).

Modes:
  model (default)  raw network mask quality per checkpoint: Dice, IoU,
                   HD/HD95, a threshold sweep, and pixel calibration (ECE, Brier).
  pipeline         end-to-end comparison of the configs in CONFIG_TYPES
                   (model + contour selection + crop) against labeled masks.
                   Tier 1 = raw mask, Tier 2 = mask after contour selection,
                   Tier 3 = crop box vs. ground truth; plus failure rates,
                   latency, and model size.
  crops            run configs on unlabeled images and save only the crops
                   plus a diagnostics manifest.

Configs:
  A  previous production pipeline: circularity>0.7 contour union,
     brightness retry only if empty, direct (warp) crop; production weights.
  B  new pipeline (select_best_contour, brightness first, squarify crop);
     production weights.
  C  same as B with SegmenterModelSkip weights; this is what
     preprocessing_images.py runs.

The pipeline primitives live in this file and are imported by
run_segmenter_inspect.py. preprocessing_images.py keeps its own copies so it
runs on its own.

All images and masks are resized to 224x224, so pixel metrics are in
224x224 units.

Default pipeline sources (each reported separately):
  old_train   mask/psd + mask/Embryo                  (original paper data)
  old_valid   mask_valid/psd + mask_valid/Embryo
  new_labels  new_labels/raw_images + new_labels/masks (Labelme, "_mask.png")
  new_valid   new_valid/raw_images + new_valid/mask_valid
Checkpoints trained on a source score optimistically on it; new_valid is
the fairest comparison (one of its images, 42256795_0397_133.9_B_0, is
also in old_train).

Usage (from preprocessing/):
    python eval_segmenter.py
    python eval_segmenter.py \\
        --weights EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt --weights-name production \\
        --weights EmbryoSegmenter/weights_skip/latest.pt --weights-name skip_connections
    python eval_segmenter.py --mode pipeline --configs A,B,C
    python eval_segmenter.py --mode pipeline --configs A,B,C \\
        --images-dir new_valid/raw_images --masks-dir new_valid/mask_valid \\
        --mask-suffix _mask --mask-ext .png --save-crops
    python eval_segmenter.py --mode crops --configs A,C \\
        --images-dir ../extracted_110-111h_focal0

Outputs (in --output-dir):
  model:    per_image_metrics.csv, threshold_sweep.csv, calibration_bins.csv,
            summary.csv, and threshold_sweep/calibration/benchmark_dice_iou/
            per_image_dice_iou .png
  pipeline: pipeline_per_image_metrics.csv, pipeline_summary.csv,
            pipeline_resources.csv, matching .png plots, and with
            --save-crops, crops/<config>/<dataset>/
  crops:    crops/<config>/<dataset>/ and crop_manifest.csv

Plots are skipped (CSVs still written) if matplotlib isn't installed.
"""

import argparse
import csv
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from EmbryoSegmenter.SegmenterModel import SegmenterModel, SegmenterModelSkip
from train_segmenter import IMAGE_EXTS, _expand_per_pair, discover_pairs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "new_labels", "raw_images")
DEFAULT_MASKS_DIR = os.path.join(SCRIPT_DIR, "new_labels", "masks")
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "EmbryoSegmenter", "weights", "epoch_102_loss_0.098710.pt")
DEFAULT_SKIP_WEIGHTS = os.path.join(SCRIPT_DIR, "EmbryoSegmenter", "weights_skip", "latest.pt")

# --mode pipeline's default sources: (name, images_dir, masks_dir, mask_suffix, mask_ext).
DEFAULT_PIPELINE_SOURCES = [
    ("old_train", os.path.join(SCRIPT_DIR, "mask", "psd"), os.path.join(SCRIPT_DIR, "mask", "Embryo"), "", ".jpg"),
    ("old_valid", os.path.join(SCRIPT_DIR, "mask_valid", "psd"), os.path.join(SCRIPT_DIR, "mask_valid", "Embryo"), "", ".jpg"),
    ("new_labels", DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, "_mask", ".png"),
    ("new_valid", os.path.join(SCRIPT_DIR, "new_valid", "raw_images"),
     os.path.join(SCRIPT_DIR, "new_valid", "mask_valid"), "_mask", ".png"),
]

# Pipeline configs, run by `run_pipeline_config()`. Add an entry to add a config.
#   weights_arg  CLI option holding the checkpoint ("prod_weights" or "skip_weights")
#   brightness   "none", "always" (correct first), or "retry_if_empty"
#                (retry on a corrected copy only if the filtered mask is empty)
#   contour      "circularity_union" or "select_best_contour"
#   crop_style   "warp" (crop the raw box) or "squarify" (square it first)
CONFIG_TYPES = {
    "A": {
        "description": "previous production pipeline: circularity>0.7 contour union, "
                       "conditional brightness-retry, direct/warp crop; production weights",
        "weights_arg": "prod_weights", 
        "brightness": "retry_if_empty",
        "contour": "circularity_union", 
        "crop_style": "warp",
    },
    "B": {
        "description": "new pipeline, production weights: "
                       "select_best_contour + always-brightness-first + squarify crop",
        "weights_arg": "prod_weights", 
        "brightness": "always",
        "contour": "select_best_contour", 
        "crop_style": "squarify",
    },
    "C": {
        "description": "new pipeline, NEW model (preprocessing_images.py): select_best_contour "
                       "+ always-brightness-first + squarify crop; SegmenterModelSkip weights",
        "weights_arg": "skip_weights", 
        "brightness": "always",
        "contour": "select_best_contour", 
        "crop_style": "squarify",
    },
}

# ---------------------------------------------------------------------------
# Pipeline primitives (also imported by run_segmenter_inspect.py)
# ---------------------------------------------------------------------------

def load_model(weights_path):
    """Load a checkpoint on CPU in eval mode, picking `SegmenterModel` or
    `SegmenterModelSkip` from deconv4's input channels (256 vs. 512)."""
    state_dict = torch.load(weights_path, map_location="cpu")
    try:
        deconv4_in_channels = state_dict["deconv4.conv.weight"].shape[1]
    except KeyError:
        raise ValueError(
            f"{weights_path} does not look like a SegmenterModel/SegmenterModelSkip "
            f"checkpoint (missing 'deconv4.conv.weight').")
    if deconv4_in_channels == 256:
        model = SegmenterModel()
    elif deconv4_in_channels == 512:
        model = SegmenterModelSkip()
    else:
        raise ValueError(
            f"{weights_path} has an unrecognized deconv4 input channel count "
            f"({deconv4_in_channels}; expected 256 for SegmenterModel or 512 for "
            f"SegmenterModelSkip).")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path, size=(224, 224)):
    """Open and resize to `size`; raw 0-255 values, not normalized."""
    img = Image.open(image_path)
    img = img.resize(size)
    return np.array(img)


def segment_image(model, frame):
    """One forward pass, thresholded at 0.5. Returns the 0.0/1.0 float mask."""
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
    """4*pi*Area/Perimeter^2 (1.0 for a perfect circle)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter ** 2))


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    """Clip `clip_hist_percent` of the darkest/lightest pixels and stretch
    the rest to [0, 255]."""
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # float(hist[i]) raises a harmless NumPy DeprecationWarning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    """(min_x, min_y, max_x, max_y) of the mask, or the full frame if empty."""
    if np.sum(mask) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    return (min_x, min_y, max_x, max_y)


def select_best_contour(mask, max_area_frac=0.9, border_margin=2, min_area_frac=0.1):
    """
    Keep a single contour instead of a circularity union:
      1. drop contours under `min_area_frac` of the frame (noise specks);
      2. prefer ones that don't touch the border (within `border_margin`)
         and aren't over `max_area_frac` of the frame (dish rim, vignette);
      3. take the largest, breaking ties by closeness to the center.
    Falls back to the full candidate list whenever a filter removes everything.
    Returns (float 0.0/1.0 mask, contour), with contour None if none found.
    """
    cv2mask = (mask * 255).astype(np.uint8)
    h, w = cv2mask.shape[:2]
    contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cv2mask)

    frame_area = float(w * h)
    frame_cx, frame_cy = w / 2.0, h / 2.0

    all_candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
    if not all_candidates:
        return (final_mask / 255.0).astype(float), None

    min_area = min_area_frac * frame_area
    candidates = [cnt for cnt in all_candidates if cv2.contourArea(cnt) >= min_area]
    if not candidates:
        candidates = all_candidates

    def touches_border(cnt):
        x, y, bw, bh = cv2.boundingRect(cnt)
        return (x <= border_margin or y <= border_margin
                or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin))

    def centroid(cnt):
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            x, y, bw, bh = cv2.boundingRect(cnt)
            return x + bw / 2.0, y + bh / 2.0
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    preferred = [
        cnt for cnt in candidates
        if not touches_border(cnt) and cv2.contourArea(cnt) <= max_area_frac * frame_area
    ]
    pool = preferred if preferred else candidates

    def score(cnt):
        area = cv2.contourArea(cnt)
        ccx, ccy = centroid(cnt)
        dist_from_center = np.hypot(ccx - frame_cx, ccy - frame_cy)
        return (area, -dist_from_center)

    best = max(pool, key=score)
    cv2.drawContours(final_mask, [best], -1, (255,), thickness=cv2.FILLED)
    return (final_mask / 255.0).astype(float), best


def squarify_bbox(min_x, min_y, max_x, max_y, img_w, img_h):
    """Grow the box to a centered square of side max(w, h), shifted to stay
    inside the image, so the 224x224 resize doesn't distort the embryo."""
    box_w = max_x - min_x
    box_h = max_y - min_y
    side = max(box_w, box_h)
    side = min(side, img_w, img_h)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    new_min_x = int(round(cx - side / 2.0))
    new_min_y = int(round(cy - side / 2.0))

    new_min_x = max(0, min(new_min_x, img_w - side))
    new_min_y = max(0, min(new_min_y, img_h - side))

    new_max_x = new_min_x + side
    new_max_y = new_min_y + side
    return int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)


def filter_mask_by_circularity_union(mask, min_circularity=0.7):
    """Config A's contour filter: union of every contour with circularity >
    `min_circularity`. Kept only for benchmarking."""
    cv2mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
    final_mask = np.zeros_like(cv2mask)
    cv2.drawContours(final_mask, circular_contours, -1, (255,), thickness=cv2.FILLED)
    return (final_mask / 255.0).astype(float)


# ---------------------------------------------------------------------------
# --mode pipeline: config runner.
# ---------------------------------------------------------------------------

def apply_contour_filter(contour, raw_mask, args):
    """Run a config's "contour" step; returns the filtered 0.0/1.0 mask."""
    if contour == "circularity_union":
        return filter_mask_by_circularity_union(raw_mask, args.circularity_threshold)
    if contour == "select_best_contour":
        filtered_mask, _selected_contour = select_best_contour(
            raw_mask, args.max_area_frac, args.border_margin, args.min_area_frac)
        return filtered_mask
    raise ValueError(f"Contour function {contour!r} not supported")


def render_pipeline_crop(img, result, out_size=(224, 224)):
    """The 224x224 crop a config would write, cut from the original (not
    brightness-adjusted) frame. "warp" slices the raw box directly;
    "squarify" squares it first."""
    min_x, min_y, max_x, max_y = result["bbox"]
    h, w = img.shape[0], img.shape[1]
    if result["crop_style"] == "squarify":
        min_x, min_y, max_x, max_y = squarify_bbox(min_x, min_y, max_x, max_y, w, h)
    crop = img[int(min_y):int(max_y), int(min_x):int(max_x)]
    if crop.size == 0:
        crop = img
    pil = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    return pil.resize(out_size)


def save_pipeline_crop(img, result, out_dir, filename):
    """Write `render_pipeline_crop` to `out_dir/filename`. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    render_pipeline_crop(img, result).save(path)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_image_mask(image_path, mask_path, size=(224, 224)):
    """--mode model loader, matching SegmentationDataset without augmentation.
    Returns the image normalized to [0, 1] and a binary mask."""
    img = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)
    img = img.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return img_arr, mask_arr


def load_gt_mask(mask_path, size=(224, 224)):
    """--mode pipeline ground truth: nearest-neighbor resize, threshold at 127."""
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.float32) > 127


def predict_prob(model, img_arr, device="cpu"):
    """Unthresholded probability map, for the threshold sweep and calibration."""
    x = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.detach().cpu().numpy()[0, 0]


def collect_labeled_pairs_with_dataset_tags(sources):
    """
    `sources`: (dataset_name, images_dir, masks_dir, mask_suffix, mask_ext)
    tuples. Returns (dataset_name, stem, image_path, mask_path) for every
    match. Unlike `discover_pairs`, shared stems are not merged, so every
    source is reported separately.
    """
    all_pairs = []
    for dataset_name, images_dir, masks_dir, mask_suffix, mask_ext in sources:
        if not os.path.isdir(images_dir):
            print(f"  WARNING: [{dataset_name}] images dir not found: {images_dir} - skipping.")
            continue
        mask_files = set(os.listdir(masks_dir)) if os.path.isdir(masks_dir) else set()
        n_found = 0
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            mask_name = f"{stem}{mask_suffix}{mask_ext}"
            if mask_name not in mask_files:
                continue
            all_pairs.append((dataset_name, stem, os.path.join(images_dir, fname),
                              os.path.join(masks_dir, mask_name)))
            n_found += 1
        print(f"  [{dataset_name}] {images_dir} + {masks_dir}: {n_found} matched pairs")
    return all_pairs


# ---------------------------------------------------------------------------
# Metrics: mask quality (shared by both modes)
# ---------------------------------------------------------------------------

def dice_iou_numpy(pred_bin, gt_bin, eps=1e-6):
    """Same formula as train_segmenter.dice_iou_at_threshold, for one numpy pair."""
    pred_bin = pred_bin.astype(np.float64).ravel()
    gt_bin = gt_bin.astype(np.float64).ravel()
    intersection = float((pred_bin * gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    union = denom - intersection
    dice = (2 * intersection + eps) / (denom + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def _border_mask(mask_bool):
    """Boundary pixels of a binary mask (mask minus its 3x3 erosion)."""
    if not mask_bool.any():
        return mask_bool.copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    border = cv2.bitwise_and(mask_u8, cv2.bitwise_not(eroded))
    return border.astype(bool)


def _distance_map_to(border_bool):
    """Per-pixel distance to the nearest True pixel of `border_bool`."""
    # cv2.distanceTransform measures distance to the nearest ZERO pixel.
    src = np.where(border_bool, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def hausdorff_distances(pred_bin, gt_bin, percentile=95.0):
    """Symmetric boundary Hausdorff distance (max) and its `percentile`-th
    percentile (HD95 by default), in pixels. (None, None) if either mask is
    empty."""
    if not pred_bin.any() or not gt_bin.any():
        return None, None

    pred_border = _border_mask(pred_bin.astype(bool))
    gt_border = _border_mask(gt_bin.astype(bool))
    if not pred_border.any():
        pred_border = pred_bin.astype(bool)
    if not gt_border.any():
        gt_border = gt_bin.astype(bool)

    dt_to_gt = _distance_map_to(gt_border)
    dt_to_pred = _distance_map_to(pred_border)

    d_pred_to_gt = dt_to_gt[pred_border]
    d_gt_to_pred = dt_to_pred[gt_border]

    hd = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))
    hd_p = float(max(np.percentile(d_pred_to_gt, percentile),
                      np.percentile(d_gt_to_pred, percentile)))
    return hd, hd_p


def reliability_diagram(all_probs, all_gts, n_bins=10):
    """Bin pixel probabilities into `n_bins`; per bin, mean prediction vs.
    fraction actually embryo. Returns (rows, ECE, Brier score)."""
    probs = all_probs.ravel()
    targets = all_gts.ravel()
    n_total = probs.size

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    ece = 0.0
    for b in range(n_bins):
        sel = bin_ids == b
        count = int(sel.sum())
        if count == 0:
            rows.append({"bin_lo": float(bin_edges[b]), "bin_hi": float(bin_edges[b + 1]),
                         "mean_pred": float("nan"), "empirical_frac": float("nan"), "count": 0})
            continue
        mean_pred = float(probs[sel].mean())
        empirical = float(targets[sel].mean())
        rows.append({"bin_lo": float(bin_edges[b]), "bin_hi": float(bin_edges[b + 1]),
                     "mean_pred": mean_pred, "empirical_frac": empirical, "count": count})
        ece += (count / n_total) * abs(mean_pred - empirical)

    brier = float(np.mean((probs - targets) ** 2))
    return rows, ece, brier


# ---------------------------------------------------------------------------
# Metrics: crop/bbox quality (--mode pipeline, Tier 3)
# ---------------------------------------------------------------------------

def bbox_iou(box_a, box_b):
    """IoU between two axis-aligned (min_x, min_y, max_x, max_y) boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def embryo_recall_in_bbox(gt_mask_bool, box):
    """Fraction of ground-truth embryo pixels inside `box` (None if GT is empty)."""
    x0, y0, x1, y1 = box
    h, w = gt_mask_bool.shape
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    total_gt = float(gt_mask_bool.sum())
    if total_gt == 0:
        return None
    inside = float(gt_mask_bool[y0:y1, x0:x1].sum())
    return inside / total_gt


def centroid_distance(box_a, box_b):
    """Euclidean distance (px) between the centers of two boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ca = ((ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0)
    cb = ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def compute_crop_diagnostics(bbox, img_w, img_h, crop_style, background_area_flag_threshold=0.7):
    """
    Failure-mode flags for a predicted box:
      crop_area_fraction                  fraction of the frame covered
      bbox_aspect_ratio                   width/height of the raw box
      touches_border                      box within 1 px of the edge
      possible_background_false_positive  area > `background_area_flag_threshold`
                                          (likely dish rim or vignette)
      likely_visible_distortion           aspect off by >20% with a "warp" crop;
                                          always False for "squarify"
    """
    min_x, min_y, max_x, max_y = bbox
    box_w = max(1, max_x - min_x)
    box_h = max(1, max_y - min_y)
    crop_area_fraction = (box_w * box_h) / float(img_w * img_h)
    aspect = box_w / box_h
    touches_border = bool(min_x <= 1 or min_y <= 1 or max_x >= img_w - 1 or max_y >= img_h - 1)
    if crop_style == "squarify":
        distorted = False
    else:
        distorted = bool(aspect > 1.2 or aspect < (1 / 1.2))
    return {
        "crop_area_fraction": crop_area_fraction,
        "bbox_aspect_ratio": aspect,
        "touches_border": touches_border,
        "possible_background_false_positive": crop_area_fraction > background_area_flag_threshold,
        "likely_visible_distortion": distorted,
    }


def evaluate_pipeline_result(config, dataset_name, stem, result, gt_bin, img_w, img_h,
                              hd_percentile, background_area_flag_threshold, latency_ms):
    """One per-image CSV row: Tier 1/2 mask metrics, Tier 3 box metrics,
    failure flags, and latency."""
    raw_bin = result["raw_mask"] > 0.5
    filtered_bin = result["filtered_mask"] > 0.5

    tier1_dice, tier1_iou = dice_iou_numpy(raw_bin, gt_bin)
    tier2_dice, tier2_iou = dice_iou_numpy(filtered_bin, gt_bin)
    tier2_hd, tier2_hd95 = hausdorff_distances(filtered_bin, gt_bin, percentile=hd_percentile)

    gt_bbox = find_bounding_box(gt_bin)
    pred_bbox = result["bbox"]
    recall = embryo_recall_in_bbox(gt_bin, pred_bbox)
    crop_diag = compute_crop_diagnostics(pred_bbox, img_w, img_h, result["crop_style"],
                                         background_area_flag_threshold)

    return {
        "dataset": dataset_name, "config": config, "image": stem,
        "tier1_dice": round(tier1_dice, 4), "tier1_iou": round(tier1_iou, 4),
        "tier2_dice": round(tier2_dice, 4), "tier2_iou": round(tier2_iou, 4),
        "tier2_hd": round(tier2_hd, 2) if tier2_hd is not None else None,
        "tier2_hd95": round(tier2_hd95, 2) if tier2_hd95 is not None else None,
        "bbox_iou": round(bbox_iou(pred_bbox, gt_bbox), 4),
        "embryo_recall_in_bbox": round(recall, 4) if recall is not None else None,
        "centroid_dist_px": round(centroid_distance(pred_bbox, gt_bbox), 2),
        "crop_area_fraction": round(crop_diag["crop_area_fraction"], 4),
        "bbox_aspect_ratio": round(crop_diag["bbox_aspect_ratio"], 4),
        "touches_border": crop_diag["touches_border"],
        "possible_background_false_positive": crop_diag["possible_background_false_positive"],
        "likely_visible_distortion": crop_diag["likely_visible_distortion"],
        "full_image_fallback_triggered": result["full_image_fallback_triggered"],
        "brightness_fallback_triggered": result["brightness_fallback_triggered"],
        "latency_ms": round(latency_ms, 3),
    }


def aggregate_pipeline_rows(rows):
    """Means, flag rates, and latency stats over a group of per-image rows."""
    def mean_of(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def rate_of(key):
        vals = [bool(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    latency_vals = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    mean_latency = float(np.mean(latency_vals)) if latency_vals else float("nan")
    std_latency = float(np.std(latency_vals, ddof=1)) if len(latency_vals) > 1 else 0.0
    p50_latency = float(np.percentile(latency_vals, 50)) if latency_vals else float("nan")
    p95_latency = float(np.percentile(latency_vals, 95)) if latency_vals else float("nan")
    throughput_ips = (1000.0 / mean_latency) if latency_vals and mean_latency > 0 else float("nan")

    return {
        "n_images": len(rows),
        "tier1_dice": mean_of("tier1_dice"), "tier1_iou": mean_of("tier1_iou"),
        "tier2_dice": mean_of("tier2_dice"), "tier2_iou": mean_of("tier2_iou"),
        "tier2_hd95": mean_of("tier2_hd95"),
        "bbox_iou": mean_of("bbox_iou"), "embryo_recall_in_bbox": mean_of("embryo_recall_in_bbox"),
        "centroid_dist_px": mean_of("centroid_dist_px"),
        "crop_area_fraction": mean_of("crop_area_fraction"),
        "pct_background_false_positive": rate_of("possible_background_false_positive"),
        "pct_touches_border": rate_of("touches_border"),
        "pct_visible_distortion": rate_of("likely_visible_distortion"),
        "pct_full_image_fallback": rate_of("full_image_fallback_triggered"),
        "pct_brightness_fallback": rate_of("brightness_fallback_triggered"),
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "throughput_ips": throughput_ips,
    }


def model_storage_stats(model, weights_path):
    """Parameter count, checkpoint file size, and in-memory size (params + buffers)."""
    n_params = sum(p.numel() for p in model.parameters())
    ram_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ram_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    weights_file_mb = os.path.getsize(weights_path) / (1024.0 ** 2)
    return {
        "weights_path": weights_path,
        "n_params": n_params,
        "n_params_m": n_params / 1e6,
        "weights_file_mb": weights_file_mb,
        "model_ram_mb": ram_bytes / (1024.0 ** 2),
    }


def build_pipeline_resource_rows(summary_all, models):
    """One time + storage row per config, from the ALL-dataset summary."""
    rows = []
    for summary in summary_all:
        config = summary["config"]
        storage = model_storage_stats(*models[config])
        rows.append({
            "config": config,
            "weights_path": storage["weights_path"],
            "n_params": storage["n_params"],
            "n_params_m": storage["n_params_m"],
            "weights_file_mb": storage["weights_file_mb"],
            "model_ram_mb": storage["model_ram_mb"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "std_latency_ms": summary.get("std_latency_ms"),
            "p50_latency_ms": summary.get("p50_latency_ms"),
            "p95_latency_ms": summary.get("p95_latency_ms"),
            "throughput_ips": summary.get("throughput_ips"),
            "n_images_timed": summary["n_images"],
        })
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        print(f"WARNING: no rows to write for {path}, skipping.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {os.path.abspath(path)}")


def round_floats(rows, ndigits=4):
    """Round every float value in a list of CSV row dicts, in place."""
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, ndigits)


def resolve_dataset_names(args):
    """--dataset-name values, or each --images-dir's basename."""
    names = _expand_per_pair(args.dataset_name, len(args.images_dir), "dataset-name", None)
    if names[0] is None:
        names = [os.path.basename(os.path.normpath(d)) for d in args.images_dir]
    return names


def default_model_name(weights_path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(weights_path)))
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    return f"{parent}/{stem}"


# ---------------------------------------------------------------------------
# Plots: --mode model
# ---------------------------------------------------------------------------

def _import_pyplot(out_path):
    """matplotlib.pyplot (Agg backend), or None with a warning if not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: matplotlib not installed, skipping {os.path.basename(out_path)} "
              "(pip install matplotlib to enable; the numbers are still in the CSVs).")
        return None
    return plt


def plot_threshold_sweep(sweep_rows, out_path):
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    models = sorted({r["model"] for r in sweep_rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, (("mean_dice", "Dice"), ("mean_iou", "IoU"), ("mean_hd95", "HD95 (px)"))):
        for model in models:
            rows = sorted((r for r in sweep_rows if r["model"] == model), key=lambda r: r["threshold"])
            xs = [r["threshold"] for r in rows]
            ys = [r[key] for r in rows]
            line, = ax.plot(xs, ys, marker=".", label=model)
            if key == "mean_dice":
                best = max(rows, key=lambda r: r["mean_dice"])
                ax.axvline(best["threshold"], color=line.get_color(), linestyle="--", alpha=0.4)
        ax.set_xlabel("threshold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote threshold-sweep plot to {os.path.abspath(out_path)} "
          f"(dashed vertical line = best mean-Dice threshold per model).")


def plot_benchmark_dice_iou(summary_rows, per_image_rows, threshold, out_path):
    """Mean Dice/IoU per model at `threshold`, with +/-1 std error bars."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    models = [r["model"] for r in summary_rows]
    n_models = len(models)
    if n_models == 0:
        return

    dice_means = [r["dice"] for r in summary_rows]
    iou_means = [r["iou"] for r in summary_rows]

    dice_stds, iou_stds = [], []
    for model in models:
        rows = [r for r in per_image_rows if r["model"] == model and r["threshold"] == threshold]
        dice_stds.append(float(np.std([r["dice"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
        iou_stds.append(float(np.std([r["iou"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)

    x = np.arange(n_models)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n_models), 5))

    bars_dice = ax.bar(x - width / 2, dice_means, width, yerr=dice_stds, capsize=4,
                       label="Dice", color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_iou = ax.bar(x + width / 2, iou_means, width, yerr=iou_stds, capsize=4,
                      label="IoU", color="#DD8452", edgecolor="white", linewidth=0.8)

    for bars, means in ((bars_dice, dice_means), (bars_iou, iou_means)):
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15 if n_models > 1 else 0, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Segmentation benchmark @ threshold={threshold:g}\n"
                 f"(bars = mean over images; error bars = ±1 std)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote benchmark Dice/IoU bar chart to {os.path.abspath(out_path)}")


def plot_per_image_dice_iou(per_image_rows, threshold, out_path):
    """Per-image Dice/IoU distributions per model (box plot + jittered points)."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    models = sorted({r["model"] for r in per_image_rows})
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 3.5 * len(models)), 5), sharey=True)
    rng = np.random.default_rng(42)

    for ax, metric, title in zip(axes, ("dice", "iou"), ("Dice", "IoU")):
        data = []
        positions = []
        for i, model in enumerate(models):
            vals = [r[metric] for r in per_image_rows
                    if r["model"] == model and r["threshold"] == threshold]
            if not vals:
                continue
            data.append(vals)
            positions.append(i)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.55,
                       color="#333333", zorder=3, edgecolors="none")

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                            showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
            for patch in bp["boxes"]:
                patch.set_facecolor("#C8D8F0" if metric == "dice" else "#F5D5C0")
                patch.set_alpha(0.85)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15 if len(models) > 1 else 0, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per image @ threshold={threshold:g}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-image score distribution (box = quartiles; dots = individual images)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per-image Dice/IoU distribution plot to {os.path.abspath(out_path)}")


def plot_calibration(calibration_curves, out_path):
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    for model_name, (bins, ece, brier) in calibration_curves.items():
        xs = [b["mean_pred"] for b in bins if b["count"] > 0]
        ys = [b["empirical_frac"] for b in bins if b["count"] > 0]
        ax.plot(xs, ys, marker="o", label=f"{model_name} (ECE={ece:.3f}, Brier={brier:.3f})")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("empirical fraction of embryo pixels")
    ax.set_title("Pixel-probability calibration (reliability diagram)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote calibration plot to {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# Plots: --mode pipeline
# ---------------------------------------------------------------------------

def plot_pipeline_dice_iou(summary_all, per_image_rows, out_path):
    """Mean Tier-2 Dice/IoU +/- std per config, all datasets combined."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    configs = [r["config"] for r in summary_all]
    if not configs:
        return
    dice_means = [r["tier2_dice"] for r in summary_all]
    iou_means = [r["tier2_iou"] for r in summary_all]
    dice_stds, iou_stds = [], []
    for cfg in configs:
        rows = [r for r in per_image_rows if r["config"] == cfg]
        dice_stds.append(float(np.std([r["tier2_dice"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
        iou_stds.append(float(np.std([r["tier2_iou"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(configs)), 5))
    bars_dice = ax.bar(x - width / 2, dice_means, width, yerr=dice_stds, capsize=4, label="Dice",
                       color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_iou = ax.bar(x + width / 2, iou_means, width, yerr=iou_stds, capsize=4, label="IoU",
                      color="#DD8452", edgecolor="white", linewidth=0.8)
    for bars, means in ((bars_dice, dice_means), (bars_iou, iou_means)):
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Pipeline benchmark: post-contour-filtering mask quality (all datasets combined)\n"
                 "(bars = mean over images; error bars = ±1 std)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote pipeline Dice/IoU bar chart to {os.path.abspath(out_path)}")


def plot_pipeline_bbox_metrics(summary_all, out_path):
    """Bbox IoU vs. GT and embryo recall-in-bbox per config."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    configs = [r["config"] for r in summary_all]
    if not configs:
        return
    bbox_iou_vals = [r["bbox_iou"] for r in summary_all]
    recall_vals = [r["embryo_recall_in_bbox"] for r in summary_all]
    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(configs)), 5))
    b1 = ax.bar(x - width / 2, bbox_iou_vals, width, label="BBox IoU vs. GT", color="#55A868",
               edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + width / 2, recall_vals, width, label="Embryo recall in BBox", color="#8172B2",
               edgecolor="white", linewidth=0.8)
    for bars, means in ((b1, bbox_iou_vals), (b2, recall_vals)):
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Pipeline benchmark: crop/bounding-box quality vs. ground truth\n"
                 "(all datasets combined)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote pipeline bbox-metrics bar chart to {os.path.abspath(out_path)}")


def plot_pipeline_failure_rates(summary_all, out_path):
    """Failure-mode rates per config (lower is better)."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    configs = [r["config"] for r in summary_all]
    if not configs:
        return
    metrics = [
        ("pct_full_image_fallback", "full-image fallback"),
        ("pct_touches_border", "bbox touches border"),
        ("pct_background_false_positive", "possible dish/vignette FP"),
        ("pct_visible_distortion", "visible aspect-ratio distortion"),
    ]
    x = np.arange(len(configs))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 2.8 * len(configs)), 5))
    colors = ["#C44E52", "#DD8452", "#CCB974", "#937860"]
    for i, (key, label) in enumerate(metrics):
        vals = [(r[key] if r[key] == r[key] else 0.0) for r in summary_all]  # NaN-safe
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width, label=label,
              color=colors[i], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("fraction of images")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pipeline benchmark: failure-mode / deployability rates\n"
                 "(all datasets combined; lower is better for all four)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote pipeline failure-rate bar chart to {os.path.abspath(out_path)}")


def plot_pipeline_per_image_box(per_image_rows, out_path):
    """Per-image Tier-2 Dice/IoU distributions per config (box plot + points)."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    configs = sorted({r["config"] for r in per_image_rows})
    if not configs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 3.2 * len(configs)), 5), sharey=True)
    rng = np.random.default_rng(42)
    for ax, metric, title in zip(axes, ("tier2_dice", "tier2_iou"), ("Dice", "IoU")):
        data, positions = [], []
        for i, cfg in enumerate(configs):
            vals = [r[metric] for r in per_image_rows if r["config"] == cfg]
            if not vals:
                continue
            data.append(vals)
            positions.append(i)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=0.5,
                       color="#333333", zorder=3, edgecolors="none")
        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                            showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
            for patch in bp["boxes"]:
                patch.set_facecolor("#C8D8F0" if metric == "tier2_dice" else "#F5D5C0")
                patch.set_alpha(0.85)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs)
        ax.set_ylabel(title)
        ax.set_title(f"{title} per image (post-contour mask)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-image pipeline score distribution, all datasets combined\n"
                "(box = quartiles; dots = individual images)", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per-image pipeline Dice/IoU distribution plot to {os.path.abspath(out_path)}")


def plot_pipeline_by_dataset(summary_by_dataset, out_path, metric="tier2_dice", title_metric="Dice"):
    """Mean `metric` per dataset source, one bar per config."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    datasets = sorted({r["dataset"] for r in summary_by_dataset})
    configs = sorted({r["config"] for r in summary_by_dataset})
    if not datasets or not configs:
        return
    x = np.arange(len(datasets))
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(datasets) * len(configs) / 2), 5))
    for i, cfg in enumerate(configs):
        vals = []
        for ds in datasets:
            match = [r[metric] for r in summary_by_dataset if r["config"] == cfg and r["dataset"] == ds]
            vals.append(match[0] if match else float("nan"))
        ax.bar(x + (i - (len(configs) - 1) / 2) * width, vals, width, label=cfg,
              edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(title_metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title_metric} by dataset source and pipeline config")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote per-dataset breakdown plot to {os.path.abspath(out_path)}")


def plot_pipeline_resources(resource_rows, out_path):
    """
    3-panel deployability figure for pipeline configs:
      1. End-to-end latency (mean +/- 1 std, ms/image, CPU)
      2. Throughput (images/sec, derived from mean latency)
      3. Model storage: parameter count (M), checkpoint file size (MB),
         in-memory footprint (MB)
    """
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    if not resource_rows:
        return

    configs = [r["config"] for r in resource_rows]
    mean_lat = [r["mean_latency_ms"] for r in resource_rows]
    std_lat = [r.get("std_latency_ms") or 0.0 for r in resource_rows]
    throughput = [r["throughput_ips"] for r in resource_rows]
    params_m = [r["n_params_m"] for r in resource_rows]
    weights_mb = [r["weights_file_mb"] for r in resource_rows]
    ram_mb = [r["model_ram_mb"] for r in resource_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(configs))

    ax = axes[0]
    bars = ax.bar(x, mean_lat, yerr=std_lat, capsize=4, color="#4C72B0",
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, mean_lat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("latency (ms / image)")
    ax.set_title("End-to-end pipeline latency\n(mean +/- std; CPU, batch size 1)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, throughput, color="#55A868", edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("throughput (images / sec)")
    ax.set_title("Derived throughput\n(1000 / mean latency ms)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    width = 0.25
    b1 = ax.bar(x - width, params_m, width, label="params (M)", color="#8172B2",
               edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x, weights_mb, width, label="checkpoint (MB)", color="#CCB974",
               edgecolor="white", linewidth=0.6)
    b3 = ax.bar(x + width, ram_mb, width, label="RAM loaded (MB)", color="#C44E52",
               edgecolor="white", linewidth=0.6)
    for bars, vals in ((b1, params_m), (b2, weights_mb), (b3, ram_mb)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("size")
    ax.set_title("Model storage / memory footprint")
    # Headroom so the legend doesn't cover the value labels.
    ax.set_ylim(0, max(max(params_m), max(weights_mb), max(ram_mb)) * 1.3)
    ax.legend(fontsize=7, loc="upper center", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Pipeline deployability: time + storage (all datasets combined)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote pipeline time/storage plot to {os.path.abspath(out_path)}")


def plot_pipeline_latency_box(per_image_rows, out_path):
    """Per-image end-to-end latency distribution per pipeline config (box plot)."""
    plt = _import_pyplot(out_path)
    if plt is None:
        return

    configs = sorted({r["config"] for r in per_image_rows})
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(configs)), 5))
    data, positions = [], []
    rng = np.random.default_rng(42)
    for i, cfg in enumerate(configs):
        vals = [r["latency_ms"] for r in per_image_rows if r["config"] == cfg]
        if not vals:
            continue
        data.append(vals)
        positions.append(i)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.45,
                   color="#333333", zorder=3, edgecolors="none")

    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor("#B8D4E8")
            patch.set_alpha(0.85)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_ylabel("latency (ms / image)")
    ax.set_title("Per-image end-to-end pipeline latency (CPU, batch size 1)\n"
                 "box = quartiles; dots = individual images")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote per-image latency distribution plot to {os.path.abspath(out_path)}")


def parse_pipeline_configs(configs_arg):
    """'A,B,C' -> ['A', 'B', 'C'], exiting on names not in CONFIG_TYPES."""
    configs = [c.strip() for c in configs_arg.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIG_TYPES)
    if unknown:
        print(f"ERROR: unknown --configs entries {sorted(unknown)}; choices are "
              f"{sorted(CONFIG_TYPES)}.", file=sys.stderr)
        sys.exit(1)
    return configs


def load_pipeline_models(args, configs):
    """{config: (model, weights_path)}; shared checkpoints are loaded once."""
    loaded = {}
    models = {}
    for config in configs:
        arg_name = CONFIG_TYPES[config]["weights_arg"]
        weights_path = getattr(args, arg_name)
        if weights_path not in loaded:
            if not os.path.exists(weights_path):
                print(f"ERROR: --{arg_name.replace('_', '-')} weights for config {config} "
                      f"not found at {weights_path}", file=sys.stderr)
                sys.exit(1)
            loaded[weights_path] = load_model(weights_path)
        models[config] = (loaded[weights_path], weights_path)
    return models


def run_pipeline_config(config, models, img, args):
    """
    Run one CONFIG_TYPES entry on `img`: brightness step, forward pass,
    contour filter, and a full-frame fallback if the mask ends up empty.
    `raw_mask` is from the pass that fed the filter (the retry, if any);
    `brightness_fallback_triggered` is None unless brightness is "retry_if_empty".
    """
    model, _weights_path = models[config]
    config_type = CONFIG_TYPES[config]
    
    brightness = config_type["brightness"]
    if brightness not in ("none", "always", "retry_if_empty"):
        raise ValueError(f"Brightness mode {brightness!r} not supported")

    seg_img = automatic_brightness_and_contrast(img) if brightness == "always" else img
    raw_mask = segment_image(model, seg_img)
    filtered_mask = apply_contour_filter(config_type["contour"], raw_mask, args)

    brightness_fallback = None
    if brightness == "retry_if_empty":
        brightness_fallback = False
        if np.max(filtered_mask) == 0:
            brightness_fallback = True
            raw_mask = segment_image(model, automatic_brightness_and_contrast(img))
            filtered_mask = apply_contour_filter(config_type["contour"], raw_mask, args)

    full_image_fallback = False
    if np.max(filtered_mask) == 0:
        full_image_fallback = True
        filtered_mask = np.ones_like(filtered_mask)
    bbox = find_bounding_box(filtered_mask)
    return {
        "raw_mask": raw_mask, "filtered_mask": filtered_mask, "bbox": bbox,
        "crop_style": config_type["crop_style"],
        "brightness_fallback_triggered": brightness_fallback,
        "full_image_fallback_triggered": full_image_fallback,
    }


def warmup_pipeline_models(models, img_arr, args, n_warmup):
    """Untimed runs so one-time warmup doesn't skew latency."""
    for config in models:
        for _ in range(n_warmup):
            run_pipeline_config(config, models, img_arr, args)


# ---------------------------------------------------------------------------
# --mode model
# ---------------------------------------------------------------------------

def run_model_mode(args):
    images_dirs = args.images_dir or [DEFAULT_IMAGES_DIR]
    masks_dirs = args.masks_dir or [DEFAULT_MASKS_DIR]

    print("Discovering (image, mask) pairs...")
    pairs_by_stem = discover_pairs(images_dirs, masks_dirs, args.mask_suffix, args.mask_ext)
    if not pairs_by_stem:
        print("ERROR: no (image, mask) pairs found. Check --images-dir/--masks-dir/"
              "--mask-suffix/--mask-ext.", file=sys.stderr)
        sys.exit(1)
    stems = sorted(pairs_by_stem)
    print(f"Evaluating on {len(stems)} labeled image/mask pair(s).\n")

    weights_paths = args.weights or [DEFAULT_WEIGHTS]
    for w in weights_paths:
        if not os.path.exists(w):
            print(f"ERROR: weights file not found at {w}", file=sys.stderr)
            sys.exit(1)
    if args.weights_name:
        if len(args.weights_name) != len(weights_paths):
            print(f"ERROR: --weights-name given {len(args.weights_name)} time(s) but "
                  f"--weights given {len(weights_paths)} time(s); pass one name per checkpoint, "
                  f"in the same order.", file=sys.stderr)
            sys.exit(1)
        weights_names = args.weights_name
    else:
        weights_names = [default_model_name(w) for w in weights_paths]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    thresholds = sorted(set(
        round(t, 4) for t in np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    ) | {round(args.threshold, 4)})

    os.makedirs(args.output_dir, exist_ok=True)

    per_image_rows = []
    sweep_rows = []
    calibration_rows = []
    summary_rows = []
    calibration_curves = {}

    for weights_path, model_name in zip(weights_paths, weights_names):
        print(f"\n=== {model_name}  ({weights_path}) ===")
        model = load_model(weights_path).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        probs_list, gts_list = [], []
        for stem in stems:
            img_path, mask_path = pairs_by_stem[stem]
            img_arr, mask_arr = load_image_mask(img_path, mask_path)
            probs_list.append(predict_prob(model, img_arr, device))
            gts_list.append(mask_arr)

        # --- 1. Dice/IoU/HD/HD95 at the fixed reporting threshold ---
        model_rows = []
        for stem, prob, gt in zip(stems, probs_list, gts_list):
            pred_bin = prob > args.threshold
            gt_bin = gt > 0.5
            dice, iou = dice_iou_numpy(pred_bin, gt_bin)
            hd, hd_p = hausdorff_distances(pred_bin, gt_bin, percentile=args.hd_percentile)
            row = {"model": model_name, "image": stem, "threshold": args.threshold,
                   "dice": round(dice, 4), "iou": round(iou, 4),
                   "hd": round(hd, 2) if hd is not None else None,
                   "hd95": round(hd_p, 2) if hd_p is not None else None}
            model_rows.append(row)
        per_image_rows.extend(model_rows)

        # --- 2. Threshold sweep ---
        model_sweep = []
        for t in thresholds:
            dices, ious, hd95s = [], [], []
            for prob, gt in zip(probs_list, gts_list):
                pred_bin = prob > t
                gt_bin = gt > 0.5
                d, i = dice_iou_numpy(pred_bin, gt_bin)
                dices.append(d)
                ious.append(i)
                _, hp = hausdorff_distances(pred_bin, gt_bin, percentile=args.hd_percentile)
                if hp is not None:
                    hd95s.append(hp)
            row = {"model": model_name, "threshold": t,
                   "mean_dice": float(np.mean(dices)), "mean_iou": float(np.mean(ious)),
                   "mean_hd95": float(np.mean(hd95s)) if hd95s else float("nan"),
                   "n_hd95_defined": len(hd95s), "n_images": len(stems)}
            model_sweep.append(row)
        sweep_rows.extend(model_sweep)

        # --- 3. Calibration (pooled pixels, threshold-independent) ---
        all_probs = np.stack(probs_list)
        all_gts = np.stack(gts_list)
        bins, ece, brier = reliability_diagram(all_probs, all_gts, n_bins=args.calibration_bins)
        calibration_curves[model_name] = (bins, ece, brier)
        for b in bins:
            calibration_rows.append({"model": model_name, **b})

        # --- Summary ---
        best = max(model_sweep, key=lambda r: r["mean_dice"])
        mean_dice = float(np.mean([r["dice"] for r in model_rows]))
        mean_iou = float(np.mean([r["iou"] for r in model_rows]))
        hd95_vals = [r["hd95"] for r in model_rows if r["hd95"] is not None]
        mean_hd95 = float(np.mean(hd95_vals)) if hd95_vals else float("nan")
        n_undefined_hd = sum(1 for r in model_rows if r["hd95"] is None)
        summary_rows.append({
            "model": model_name, "weights_path": weights_path, "n_images": len(stems),
            "n_params": n_params, "reporting_threshold": args.threshold,
            "dice": round(mean_dice, 4), "iou": round(mean_iou, 4),
            "hd95": round(mean_hd95, 2) if hd95_vals else None,
            "n_images_with_undefined_hd": n_undefined_hd,
            "best_threshold_by_dice": best["threshold"],
            "dice_at_best_threshold": round(best["mean_dice"], 4),
            "iou_at_best_threshold": round(best["mean_iou"], 4),
            "ece": round(ece, 4), "brier": round(brier, 4),
        })

        print(f"  n_params={n_params:,}")
        print(f"  @threshold={args.threshold}: Dice={mean_dice:.4f}  IoU={mean_iou:.4f}  "
              f"HD95={mean_hd95:.2f}px" + (f"  ({n_undefined_hd} image(s) had no valid mask "
              f"at this threshold, excluded from HD95)" if n_undefined_hd else ""))
        print(f"  Best threshold by mean Dice: {best['threshold']:.2f}  "
              f"(Dice={best['mean_dice']:.4f}, IoU={best['mean_iou']:.4f}, "
              f"HD95={best['mean_hd95']:.2f}px)")
        print(f"  Calibration: ECE={ece:.4f}  Brier={brier:.4f}  "
              f"(lower is better for both; ECE=0 means predicted probabilities exactly "
              f"match empirical pixel frequencies)")

    write_csv(os.path.join(args.output_dir, "per_image_metrics.csv"), per_image_rows)
    write_csv(os.path.join(args.output_dir, "threshold_sweep.csv"), sweep_rows)
    write_csv(os.path.join(args.output_dir, "calibration_bins.csv"), calibration_rows)
    write_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows)
    plot_threshold_sweep(sweep_rows, os.path.join(args.output_dir, "threshold_sweep.png"))
    plot_calibration(calibration_curves, os.path.join(args.output_dir, "calibration.png"))
    plot_benchmark_dice_iou(summary_rows, per_image_rows, args.threshold,
                            os.path.join(args.output_dir, "benchmark_dice_iou.png"))
    plot_per_image_dice_iou(per_image_rows, args.threshold,
                            os.path.join(args.output_dir, "per_image_dice_iou.png"))

    print(f"\nDone. All outputs written to {os.path.abspath(args.output_dir)}.")
    print("See summary.csv for the headline numbers, benchmark_dice_iou.png for the "
          "standard mean±std comparison figure, per_image_dice_iou.png for per-image "
          "spread, threshold_sweep.png to check whether 0.5 is actually the best cutoff, "
          "and calibration.png for pixel-probability reliability.")


# ---------------------------------------------------------------------------
# --mode pipeline
# ---------------------------------------------------------------------------

def run_pipeline_mode(args):
    if args.images_dir:
        n_pairs = len(args.images_dir)
        if not args.masks_dir or len(args.masks_dir) != n_pairs:
            print("ERROR: --masks-dir must be given the same number of times as --images-dir.",
                  file=sys.stderr)
            sys.exit(1)
        mask_suffixes = _expand_per_pair(args.mask_suffix, n_pairs, "mask-suffix", "_mask")
        mask_exts = _expand_per_pair(args.mask_ext, n_pairs, "mask-ext", ".png")
        sources = list(zip(resolve_dataset_names(args), args.images_dir, args.masks_dir,
                           mask_suffixes, mask_exts))
    else:
        sources = DEFAULT_PIPELINE_SOURCES
        print("No --images-dir given; using this repo's 4 default labeled sources "
              "(old paper train + old paper val + new Labelme labels + new Labelme val):")

    print("Discovering labeled (image, mask) pairs across all dataset sources...")
    pairs = collect_labeled_pairs_with_dataset_tags(sources)
    if not pairs:
        print("ERROR: no (image, mask) pairs found across any source.", file=sys.stderr)
        sys.exit(1)
    print(f"Total labeled pairs across all sources: {len(pairs)}\n")

    configs_to_run = parse_pipeline_configs(args.configs)
    models = load_pipeline_models(args, configs_to_run)
    n_params_by_config = {c: sum(p.numel() for p in model.parameters())
                          for c, (model, _weights_path) in models.items()}

    print("Running configs (CPU-only, matching production/run_segmenter_inspect.py):")
    for c in configs_to_run:
        print(f"  {c}: {CONFIG_TYPES[c]['description']}")
    print()

    if args.timing_warmup > 0 and pairs:
        _, _, warmup_path, _ = pairs[0]
        warmup_img = load_image(warmup_path)
        print(f"Timing warmup: {args.timing_warmup} untimed iteration(s) per config "
              f"on {os.path.basename(warmup_path)} ...")
        warmup_pipeline_models(models, warmup_img, args, args.timing_warmup)
        print()

    per_image_rows = []
    n_crops_saved = 0
    for dataset_name, stem, img_path, mask_path in pairs:
        img_arr = load_image(img_path)
        img_h, img_w = img_arr.shape[0], img_arr.shape[1]
        gt_bin = load_gt_mask(mask_path, size=(img_w, img_h))

        for config in configs_to_run:
            t0 = time.perf_counter()
            result = run_pipeline_config(config, models, img_arr, args)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            row = evaluate_pipeline_result(config, dataset_name, stem, result, gt_bin, img_w, img_h,
                                           args.hd_percentile, args.background_area_flag_threshold,
                                           latency_ms)
            per_image_rows.append(row)

            if args.save_crops:
                crop_dir = os.path.join(args.output_dir, "crops", config, dataset_name)
                save_pipeline_crop(img_arr, result, crop_dir, os.path.basename(img_path))
                n_crops_saved += 1

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "pipeline_per_image_metrics.csv"), per_image_rows)

    dataset_names_used = sorted({r["dataset"] for r in per_image_rows})

    summary_rows = []
    for config in configs_to_run:
        combined_rows = [r for r in per_image_rows if r["config"] == config]
        agg = aggregate_pipeline_rows(combined_rows)
        summary_rows.append({"dataset": "ALL", "config": config,
                             "n_params": n_params_by_config[config], **agg})
        for dataset_name in dataset_names_used:
            ds_rows = [r for r in combined_rows if r["dataset"] == dataset_name]
            if not ds_rows:
                continue
            agg_ds = aggregate_pipeline_rows(ds_rows)
            summary_rows.append({"dataset": dataset_name, "config": config,
                                 "n_params": n_params_by_config[config], **agg_ds})

    round_floats(summary_rows)
    write_csv(os.path.join(args.output_dir, "pipeline_summary.csv"), summary_rows)

    summary_all = [r for r in summary_rows if r["dataset"] == "ALL"]
    summary_by_dataset = [r for r in summary_rows if r["dataset"] != "ALL"]

    print("\n=== Pipeline benchmark summary (all datasets combined) ===")
    for r in summary_all:
        print(f"  [{r['config']}] n={r['n_images']}  Tier2 Dice={r['tier2_dice']:.4f}  "
              f"IoU={r['tier2_iou']:.4f}  HD95={r['tier2_hd95']:.2f}px  "
              f"BBoxIoU={r['bbox_iou']:.4f}  Recall={r['embryo_recall_in_bbox']:.4f}  "
              f"latency={r['mean_latency_ms']:.2f}ms (p95={r['p95_latency_ms']:.2f})  "
              f"throughput={r['throughput_ips']:.1f} img/s  "
              f"fallback={r['pct_full_image_fallback']:.2%}  border={r['pct_touches_border']:.2%}  "
              f"bg_fp={r['pct_background_false_positive']:.2%}  "
              f"distortion={r['pct_visible_distortion']:.2%}")

    resource_rows = build_pipeline_resource_rows(summary_all, models)
    round_floats(resource_rows)
    write_csv(os.path.join(args.output_dir, "pipeline_resources.csv"), resource_rows)

    print("\n=== Pipeline deployability (time + storage) ===")
    for r in resource_rows:
        print(f"  [{r['config']}] params={r['n_params']:,}  "
              f"checkpoint={r['weights_file_mb']:.2f} MB  RAM={r['model_ram_mb']:.2f} MB  "
              f"latency mean={r['mean_latency_ms']:.2f}ms p50={r['p50_latency_ms']:.2f}ms "
              f"p95={r['p95_latency_ms']:.2f}ms  throughput={r['throughput_ips']:.1f} img/s")

    plot_pipeline_dice_iou(summary_all, per_image_rows,
                           os.path.join(args.output_dir, "pipeline_dice_iou.png"))
    plot_pipeline_bbox_metrics(summary_all, os.path.join(args.output_dir, "pipeline_bbox_metrics.png"))
    plot_pipeline_failure_rates(summary_all, os.path.join(args.output_dir, "pipeline_failure_rates.png"))
    plot_pipeline_per_image_box(per_image_rows,
                                os.path.join(args.output_dir, "pipeline_per_image_dice_iou.png"))
    plot_pipeline_by_dataset(summary_by_dataset, os.path.join(args.output_dir, "pipeline_dice_by_dataset.png"),
                             metric="tier2_dice", title_metric="Dice")
    plot_pipeline_resources(resource_rows, os.path.join(args.output_dir, "pipeline_resources.png"))
    plot_pipeline_latency_box(per_image_rows, os.path.join(args.output_dir, "pipeline_latency_box.png"))

    print(f"\nDone. All outputs written to {os.path.abspath(args.output_dir)}.")
    if args.save_crops:
        print(f"Saved {n_crops_saved} cropped images under "
              f"{os.path.abspath(os.path.join(args.output_dir, 'crops'))} "
              f"(one folder per config: {', '.join(configs_to_run)}).")
    print("Deployability: see pipeline_resources.csv, pipeline_resources.png, "
          "and pipeline_latency_box.png.")
    if "old_train" in dataset_names_used or "old_valid" in dataset_names_used:
        print("\nCAUTION: the 'old_train'/'old_valid' sources (mask/psd+Embryo, "
              "mask_valid/psd+Embryo) are the original paper's labeled dataset. If EITHER "
              "checkpoint being evaluated was itself trained/fine-tuned on some of these "
              "images (check what --images-dir/--masks-dir you used with train_segmenter.py "
              "to produce it), metrics on that source are optimistically biased for that "
              "checkpoint. Compare the per-dataset breakdown in pipeline_summary.csv "
              "(dataset != 'ALL') and pipeline_dice_by_dataset.png rather than trusting the "
              "combined 'ALL' number alone.")


# ---------------------------------------------------------------------------
# --mode crops
# ---------------------------------------------------------------------------

def run_crops_mode(args):
    """Save each config's 224x224 crops for unlabeled images, plus a
    diagnostics manifest. Blank images are skipped."""
    if not args.images_dir:
        print("ERROR: --mode crops needs at least one --images-dir.", file=sys.stderr)
        sys.exit(1)
    dataset_names = resolve_dataset_names(args)

    configs_to_run = parse_pipeline_configs(args.configs)
    models = load_pipeline_models(args, configs_to_run)
    print("Running configs (CPU-only):")
    for c in configs_to_run:
        print(f"  {c}: {CONFIG_TYPES[c]['description']}")

    crops_root = os.path.join(args.output_dir, "crops")
    manifest_rows = []
    n_blank = 0
    t_start = time.perf_counter()

    for dataset_name, images_dir in zip(dataset_names, args.images_dir):
        if not os.path.isdir(images_dir):
            print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
            sys.exit(1)
        fnames = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS))
        print(f"\n[{dataset_name}] {len(fnames)} images in {images_dir}")

        for i, fname in enumerate(fnames, 1):
            img_arr = load_image(os.path.join(images_dir, fname))
            if np.max(img_arr) == 0:
                n_blank += 1
                continue
            img_h, img_w = img_arr.shape[0], img_arr.shape[1]

            for config in configs_to_run:
                result = run_pipeline_config(config, models, img_arr, args)
                out_dir = os.path.join(crops_root, config, dataset_name)
                crop_path = save_pipeline_crop(img_arr, result, out_dir, fname)
                diag = compute_crop_diagnostics(result["bbox"], img_w, img_h, result["crop_style"],
                                                args.background_area_flag_threshold)
                manifest_rows.append({
                    "dataset": dataset_name, "config": config, "image": fname,
                    "crop_path": crop_path,
                    "bbox_xyxy": " ".join(str(int(v)) for v in result["bbox"]),
                    "crop_area_fraction": round(diag["crop_area_fraction"], 4),
                    "bbox_aspect_ratio": round(diag["bbox_aspect_ratio"], 4),
                    "touches_border": diag["touches_border"],
                    "possible_background_false_positive": diag["possible_background_false_positive"],
                    "likely_visible_distortion": diag["likely_visible_distortion"],
                    "full_image_fallback_triggered": result["full_image_fallback_triggered"],
                    "brightness_fallback_triggered": result["brightness_fallback_triggered"],
                })

            if i % 250 == 0 or i == len(fnames):
                elapsed = time.perf_counter() - t_start
                print(f"  [{dataset_name}] {i}/{len(fnames)} images ({elapsed / 60:.1f} min elapsed)")

    if not manifest_rows:
        print("ERROR: no non-blank images found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "crop_manifest.csv")
    write_csv(manifest_path, manifest_rows)

    print("\n=== Crop summary per config ===")
    for config in configs_to_run:
        agg = aggregate_pipeline_rows([r for r in manifest_rows if r["config"] == config])
        print(f"  [{config}] n={agg['n_images']}  "
              f"fallback={agg['pct_full_image_fallback']:.2%}  "
              f"border={agg['pct_touches_border']:.2%}  "
              f"bg_fp={agg['pct_background_false_positive']:.2%}  "
              f"distortion={agg['pct_visible_distortion']:.2%}  "
              f"mean crop area={agg['crop_area_fraction']:.3f}")
    if n_blank:
        print(f"Skipped {n_blank} blank image(s).")
    print(f"\nDone in {(time.perf_counter() - t_start) / 60:.1f} min. Crops under "
          f"{os.path.abspath(crops_root)}/<config>/<dataset>/; diagnostics in "
          f"{os.path.abspath(manifest_path)}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the embryo segmenter: raw mask quality (--mode model), "
                    "end-to-end pipeline configs (--mode pipeline), or crops only "
                    "(--mode crops). See the module docstring.")
    parser.add_argument("--mode", choices=["model", "pipeline", "crops"], default="model",
                        help="model (default), pipeline, or crops.")

    data = parser.add_argument_group("data selection")
    data.add_argument("--images-dir", action="append", default=None,
                      help="Raw image directory (repeatable). Model-mode default: "
                           f"{DEFAULT_IMAGES_DIR}; pipeline-mode default: the 4 labeled sources.")
    data.add_argument("--masks-dir", action="append", default=None,
                      help="Mask directory for each --images-dir (repeatable). "
                           f"Model-mode default: {DEFAULT_MASKS_DIR}.")
    data.add_argument("--mask-suffix", action="append", default=None,
                      help="Mask filename suffix per pair (default: _mask). Use '' with "
                           "--mask-ext .jpg for the original paper dataset.")
    data.add_argument("--mask-ext", action="append", default=None,
                      help="Mask file extension per pair (default: .png).")
    data.add_argument("--dataset-name", action="append", default=None,
                      help="Display name per --images-dir (default: directory basename).")

    model_grp = parser.add_argument_group("--mode model options")
    model_grp.add_argument("--weights", action="append", default=None,
                           help=f"Checkpoint to evaluate (repeatable; default: {DEFAULT_WEIGHTS}). "
                                "Architecture is auto-detected.")
    model_grp.add_argument("--weights-name", action="append", default=None,
                           help="Display name per --weights, in order (default: parent-dir/filename).")
    model_grp.add_argument("--threshold", type=float, default=0.5,
                           help="Reporting threshold (default: 0.5).")
    model_grp.add_argument("--threshold-min", type=float, default=0.05,
                           help="Lowest threshold in the sweep (default: 0.05).")
    model_grp.add_argument("--threshold-max", type=float, default=0.95,
                           help="Highest threshold in the sweep (default: 0.95).")
    model_grp.add_argument("--threshold-step", type=float, default=0.05,
                           help="Sweep step (default: 0.05).")
    model_grp.add_argument("--calibration-bins", type=int, default=10,
                           help="Probability bins for calibration/ECE (default: 10).")

    pipeline_grp = parser.add_argument_group("--mode pipeline options")
    pipeline_grp.add_argument("--prod-weights", default=DEFAULT_WEIGHTS,
                              help=f"SegmenterModel checkpoint for configs A/B (default: {DEFAULT_WEIGHTS}).")
    pipeline_grp.add_argument("--skip-weights", default=DEFAULT_SKIP_WEIGHTS,
                              help="SegmenterModelSkip checkpoint for config C "
                                   f"(default: {DEFAULT_SKIP_WEIGHTS}).")
    pipeline_grp.add_argument("--configs", default="A,B,C",
                              help="Comma-separated configs from CONFIG_TYPES (default: A,B,C).")
    pipeline_grp.add_argument("--max-area-frac", type=float, default=0.9,
                              help="select_best_contour: max contour area as a frame fraction (default: 0.9).")
    pipeline_grp.add_argument("--border-margin", type=int, default=2,
                              help="select_best_contour: border-touch margin in px (default: 2).")
    pipeline_grp.add_argument("--min-area-frac", type=float, default=0.1,
                              help="select_best_contour: min contour area as a frame fraction (default: 0.1).")
    pipeline_grp.add_argument("--circularity-threshold", type=float, default=0.7,
                              help="Config A circularity threshold (default: 0.7).")
    pipeline_grp.add_argument("--background-area-flag-threshold", type=float, default=0.7,
                              help="Flag crops covering more than this frame fraction (default: 0.7).")
    pipeline_grp.add_argument("--timing-warmup", type=int, default=3,
                              help="Untimed iterations per model before timing (default: 3; 0 disables).")
    pipeline_grp.add_argument("--save-crops", action="store_true",
                              help="Also save each config's crops to <output-dir>/crops/<config>/<dataset>/.")

    shared = parser.add_argument_group("shared options")
    shared.add_argument("--hd-percentile", type=float, default=95.0,
                        help="Percentile for the hd95 metric (default: 95; 100 = max Hausdorff).")
    shared.add_argument("--output-dir", default=None,
                        help="Output directory (default: segmenter_eval_output, "
                             "segmenter_pipeline_eval_output, or segmenter_crops_output).")
    shared.add_argument("--device", default=None,
                        help="cuda/cpu for --mode model (default: auto). Pipeline modes run on CPU.")

    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = {"model": "segmenter_eval_output",
                           "pipeline": "segmenter_pipeline_eval_output",
                           "crops": "segmenter_crops_output"}[args.mode]

    if args.mode == "model":
        run_model_mode(args)
    elif args.mode == "pipeline":
        run_pipeline_mode(args)
    else:
        run_crops_mode(args)


if __name__ == "__main__":
    main()
