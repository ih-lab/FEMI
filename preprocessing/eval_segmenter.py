"""
eval_segmenter.py

Quantitative benchmarking for `EmbryoSegmenter/SegmenterModel.py` (and its
`SegmenterModelSkip` U-Net variant from `train_segmenter.py`), at two levels:

  --mode model (default): raw CNN mask quality for one or more checkpoints -
      Dice, IoU, Hausdorff/HD95, a probability-threshold sweep, and
      pixel-probability calibration. This is the "is the network's output
      itself accurate" question - see the original module docstring section
      below for detail. Unchanged from this script's first version.

  --mode pipeline: full END-TO-END preprocessing-pipeline comparison - model
      + contour-selection/fallback logic + final crop geometry - across
      named pipeline "configs" that mirror this repo's actual, divergent
      implementations:
        A       production today (`preprocessing_images.py`): circularity>0.7
                contour union, conditional brightness-retry, direct/"warp" crop.
        Bprime  ablation: the new `select_best_contour` logic (no circularity,
                always-brightness-first) with production's OLD warp crop -
                isolates the contour-selection fix from the crop-squarify fix.
        B       the full new pipeline (`run_segmenter_inspect.py`) with the
                CURRENT production weights: new contour logic + squarify crop.
        C       the full new pipeline with the NEW `SegmenterModelSkip`
                weights (`train_segmenter.py --skip-connections`).
      Reports mask quality (Tier 1: raw network output; Tier 2: after
      contour-selection - the mask actually used to crop) AND crop-level
      metrics (Tier 3: bounding-box IoU vs. ground truth, embryo recall
      inside the predicted box, centroid error, crop-area fraction) plus
      failure-mode/deployability rates (full-image fallback, border-touch,
      possible dish/vignette false positive, visible aspect-ratio
      distortion), per-image end-to-end latency (mean/std/p50/p95,
      throughput), and model storage (parameter count, checkpoint file
      size, in-memory RAM footprint) - see `run_pipeline_mode()` below.

Both modes share the same low-level pipeline primitives, which now live
ONLY in this file (moved from `run_segmenter_inspect.py`, which imports them
back - see that file's updated docstring): `load_model`, `load_image`,
`segment_image`, `calculate_circularity`, `automatic_brightness_and_contrast`,
`find_bounding_box`, `select_best_contour`, `squarify_bbox`. Config A's
circularity-union filter (`filter_mask_by_circularity_union`) is ALSO
duplicated here from `preprocessing_images.py`, for the same reason
`run_segmenter_inspect.py` already duplicates (rather than imports) that
file's logic: `preprocessing_images.py` runs model-loading/directory-listing/
`ProcessPoolExecutor` as top-level side effects at import time and uses
placeholder paths, so it is not safely importable - see that file's module
docstring.

Reuses this repo's existing conventions rather than reinventing them:
  - Pair discovery (--mode model): `discover_pairs()` from `train_segmenter.py`
    (same filename-stem matching, same `--mask-suffix`/`--mask-ext` semantics).
  - Pair discovery (--mode pipeline): a dataset-tagged variant of the same
    idea (`collect_labeled_pairs_with_dataset_tags`) that keeps every
    source's pairs distinct instead of merging/overriding by stem, since for
    evaluation (unlike training) we want to report on every source
    separately, not silently drop overlapping stems.
  - Image/mask resizing: same as `SegmentationDataset` in `train_segmenter.py`
    (image: bilinear, mask: nearest, resized together to the model's fixed
    224x224 input/output size). All pixel-distance metrics (Hausdorff/HD95,
    bbox coordinates, centroid distance) are therefore in 224x224-pixel
    units, not the original annotated image's resolution.
  - Dice/IoU formula: identical to `dice_iou_at_threshold()` in
    `train_segmenter.py` (same epsilon, same definition).

--mode pipeline data sources: by default, ALL THREE of this repo's labeled
sets, tagged by dataset in the output so you can check whether a config's
ranking holds up consistently across sources (and so you can spot
optimistic bias if a checkpoint was trained on one of them - see the
caution message printed at the end of a pipeline run):
  old_train   preprocessing/mask/psd (raw) + preprocessing/mask/Embryo (mask)
              - the original paper's 30-pair dataset (mask named identically
              to the image, e.g. "foo.jpg" raw + "foo.jpg" mask).
  old_valid   preprocessing/mask_valid/psd + preprocessing/mask_valid/Embryo
              - that dataset's held-out validation split.
  new_labels  preprocessing/new_labels/raw_images + preprocessing/new_labels/masks
              - this repo's newer Labelme-annotated set
              (labelme_json_to_mask.py convention: "foo.jpg" + "foo_mask.png").

Usage:
    cd preprocessing

    # --mode model (default): evaluate production weights on new_labels/:
    python eval_segmenter.py

    # --mode model: compare production vs. a fine-tuned/skip-connection checkpoint:
    python eval_segmenter.py \\
        --weights EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt --weights-name production \\
        --weights EmbryoSegmenter/weights_skip/latest.pt --weights-name skip_connections

    # --mode pipeline: benchmark all 4 configs (A, Bprime, B, C) across all 3
    # default labeled sources:
    python eval_segmenter.py --mode pipeline

    # --mode pipeline: only the 3 "real" configs the fixes are meant to compare
    # (skip the Bprime ablation), on just the new_labels/ set:
    python eval_segmenter.py --mode pipeline --configs A,B,C \\
        --images-dir new_labels/raw_images --masks-dir new_labels/masks \\
        --mask-suffix _mask --mask-ext .png

Outputs (written to --output-dir):
  --mode model (default dir: ./segmenter_eval_output):
    per_image_metrics.csv   one row per (model, image): dice, iou, hd, hd95 @ --threshold
    threshold_sweep.csv     one row per (model, threshold): mean dice/iou/hd95
    calibration_bins.csv    one row per (model, probability bin): mean_pred, empirical_frac, count
    summary.csv             one row per model: headline numbers @ --threshold, best threshold, ECE, Brier
    threshold_sweep.png     Dice/IoU/HD95 vs. threshold, one line per model
    calibration.png         reliability diagram, one curve per model
    benchmark_dice_iou.png  headline benchmark figure: mean Dice/IoU +/- std per model (bar chart)
    per_image_dice_iou.png  per-image Dice/IoU distributions per model (box plot)

  --mode pipeline (default dir: ./segmenter_pipeline_eval_output):
    pipeline_per_image_metrics.csv  one row per (dataset, config, image): all tiers
    pipeline_summary.csv            one row per (dataset, config) incl. "ALL" (combined)
    pipeline_dice_iou.png           Tier-2 (post-contour) Dice/IoU +/- std per config
    pipeline_bbox_metrics.png       bbox IoU + embryo recall-in-bbox per config
    pipeline_failure_rates.png      fallback/border-touch/bg-FP/distortion rates per config
    pipeline_per_image_dice_iou.png per-image Tier-2 Dice/IoU distributions (box plot)
    pipeline_dice_by_dataset.png    Tier-2 Dice per config, broken out by dataset source
    pipeline_resources.csv          one row per config: timing (mean/std/p50/p95, throughput)
                                    + storage (parameter count, checkpoint file size MB,
                                    in-memory model footprint MB)
    pipeline_resources.png          3-panel deployability figure: latency, throughput,
                                    model storage (params + checkpoint + RAM)
    pipeline_latency_box.png        per-image end-to-end latency distribution per config

No new dependencies: only numpy/torch/opencv/pillow (already required by
EmbryoSegmenter/requirements-segmenter.txt) plus matplotlib for plots (also
already listed there; if missing, CSVs are still written and only plots
are skipped).
"""

import argparse
import csv
import os
import sys
import time
import warnings

# automatic_brightness_and_contrast() below (copied verbatim from
# preprocessing_images.py for behavioral fidelity) triggers a harmless
# NumPy scalar-conversion DeprecationWarning; silenced here for cleaner
# console output. This does not affect any computed values. (Moved here
# from run_segmenter_inspect.py along with the function itself.)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import torch
from PIL import Image

from EmbryoSegmenter.SegmenterModel import SegmenterModel
from train_segmenter import IMAGE_EXTS, SegmenterModelSkip, _expand_per_pair, discover_pairs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "new_labels", "raw_images")
DEFAULT_MASKS_DIR = os.path.join(SCRIPT_DIR, "new_labels", "masks")
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "EmbryoSegmenter", "weights", "epoch_102_loss_0.098710.pt")
DEFAULT_SKIP_WEIGHTS = os.path.join(SCRIPT_DIR, "EmbryoSegmenter", "weights_skip", "latest.pt")

# --mode pipeline's default data sources - see module docstring.
DEFAULT_PIPELINE_SOURCES = [
    ("old_train", os.path.join(SCRIPT_DIR, "mask", "psd"), os.path.join(SCRIPT_DIR, "mask", "Embryo"), "", ".jpg"),
    ("old_valid", os.path.join(SCRIPT_DIR, "mask_valid", "psd"), os.path.join(SCRIPT_DIR, "mask_valid", "Embryo"), "", ".jpg"),
    ("new_labels", DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, "_mask", ".png"),
]

CONFIG_DESCRIPTIONS = {
    "A": "production today (preprocessing_images.py): circularity>0.7 contour union, "
         "conditional brightness-retry, direct/warp crop; production weights",
    "Bprime": "ablation: select_best_contour (no circularity, always-brightness-first) "
              "+ OLD warp crop - isolates the contour fix from the crop-squarify fix; "
              "production weights",
    "B": "new pipeline (run_segmenter_inspect.py), OLD model: select_best_contour "
         "+ always-brightness-first + squarify crop; production weights",
    "C": "new pipeline, NEW model: select_best_contour + always-brightness-first "
         "+ squarify crop; SegmenterModelSkip weights",
}


# ---------------------------------------------------------------------------
# Shared low-level pipeline primitives (moved from run_segmenter_inspect.py -
# see that file's updated docstring. Behavior is unchanged; only the module
# they live in changed).
# ---------------------------------------------------------------------------

def load_model(weights_path):
    """
    Load a segmenter checkpoint, auto-detecting whether it's the production
    `SegmenterModel` architecture or the `SegmenterModelSkip` (U-Net, skip
    connections) variant trained via `train_segmenter.py --skip-connections`
    - see that file for the SegmenterModelSkip class itself
    (`EmbryoSegmenter/SegmenterModel.py` has no skip-connection variant, by
    design). The two are distinguished by `deconv4.conv.weight`'s input
    channel count (256 = no skip, 512 = skip, since skip connections
    concatenate deconv5's 256-channel output with conv5's 256-channel
    feature map). Auto-detecting means callers don't need to track which
    architecture a given .pt file was trained with. Always loads to CPU
    (map_location="cpu") and calls .eval() - matches production and
    run_segmenter_inspect.py, which never move this model off CPU.
    """
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
    """Same behavior as `load_image` in preprocessing_images.py: open,
    resize to a fixed size, and convert to a numpy array (whatever mode
    the source image is in - grayscale or RGB), with RAW pixel values
    (0-255ish), NOT normalized. This is the convention `segment_image()`/
    `automatic_brightness_and_contrast()` below expect."""
    img = Image.open(image_path)
    img = img.resize(size)
    return np.array(img)


def segment_image(model, frame):
    """Same behavior as `segment_image` in preprocessing_images.py: one
    forward pass through SegmenterModel/SegmenterModelSkip, thresholded at
    0.5. Returns just the binary (0.0/1.0 float) mask - this repo's
    production code never uses the discarded pixel-masked
    `segmented_images` output either (see SEGMENTER_ACCURACY_ROADMAP.md
    Section 1)."""
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
    """Same formula as preprocessing_images.py: 4*pi*Area/Perimeter^2 (1.0
    for a perfect circle). Used as a hard filter by config A
    (`filter_mask_by_circularity_union`) and purely as a diagnostic for
    configs B/B'/C (which no longer use it to select/reject contours)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter ** 2))


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    """Identical to preprocessing_images.py: clip a percentage of the
    darkest/lightest pixels (via the cumulative grayscale histogram) and
    linearly stretch the remaining range to fill [0, 255]."""
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
    """Identical to preprocessing_images.py: returns (min_x, min_y, max_x,
    max_y), or the full-frame box if the mask has no foreground pixels."""
    if np.sum(mask) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    return (min_x, min_y, max_x, max_y)


def select_best_contour(mask, max_area_frac=0.9, border_margin=2, min_area_frac=0.1):
    """
    Fixed replacement for production's circularity-union filter (root
    causes A/B in SEGMENTER_ACCURACY_ROADMAP.md; see
    `filter_mask_by_circularity_union` below for the production logic this
    replaces). Instead of taking the UNION of every contour whose
    *circularity* clears a hard threshold, this picks a SINGLE
    best-candidate contour, and does not use circularity at all:

      0. Discard contours smaller than `min_area_frac` of the frame before
         anything else (see "noise-speck bug" note below) - these are
         essentially always thresholding noise, never a real embryo.
      1. Find all remaining external contours in the raw thresholded mask.
      2. Prefer contours that (a) do NOT touch/nearly touch the image
         border, and (b) are not implausibly large (> `max_area_frac` of
         the frame). A petri dish rim or the microscope's light vignette
         almost always trips at least one of these checks; a real embryo
         (even off-center) usually trips neither. Unlike circularity, this
         makes no assumption about the embryo's shape, so it works equally
         well for round, hatching, or multi-cell embryos.
      3. Among the surviving candidates (or, if none survive, among ALL
         candidates - border-touching is a strong signal but not treated
         as an absolute veto, since a genuinely off-center/edge-framed
         embryo should still be picked over nothing), choose the one with
         the LARGEST area, breaking ties by proximity to the frame center.
      4. Draw ONLY that one contour (filled) onto the output mask - no
         union with any other contour.

    Noise-speck bug (found while comparing SegmenterModelSkip against
    production on 92442871_0393_134.8_HB_0.jpg): step 0 didn't originally
    exist. Without it, if the raw mask contains ANY tiny spurious blob that
    happens to not touch the border, step 2's "prefer border-safe
    candidates" rule made that speck the entire `preferred` pool, so it won
    unconditionally over the real embryo even when the real embryo was
    5000x larger and only disqualified for grazing the frame edge by 1-2
    pixels. Filtering by minimum area FIRST closes this.

    `min_area_frac` default (0.1, i.e. 10%): this now matches everywhere
    it's used in this repo (this function's own default, the
    run_segmenter_inspect.py CLI default, and every --mode pipeline config
    below) - a previous version of the run_segmenter_inspect.py CLI
    default was inconsistently set to 0.01 despite its own help text
    already claiming 0.1; that mismatch has been fixed.

    Returns (final_mask, selected_contour) where final_mask matches the
    input mask's shape/dtype convention (float, 0.0/1.0) and
    selected_contour is None if no contour was found at all.
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
    """
    Expand a (possibly non-square) bounding box into a square box of side
    `max(width, height)`, centered on the original box's center, then
    shift it (without resizing) so it stays fully within the image bounds.

    Fixes a real distortion bug in production (`preprocessing_images.py`
    crops directly to the mask's often-non-square bounding box, then does
    `Image.resize((224, 224))`, which does NOT preserve aspect ratio - see
    root cause F in SEGMENTER_ACCURACY_ROADMAP.md). Cropping a SQUARE
    region first means the final resize is uniform, with no distortion.
    """
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
    """
    Faithful reimplementation of the contour-filtering logic in
    `preprocessing_images.py`'s `process_image()` - production's ACTUAL
    current behavior ("config A" below): keep every contour whose
    circularity exceeds `min_circularity` and draw the UNION of all of
    them (as opposed to `select_best_contour`'s single-best-candidate,
    no-circularity approach). Duplicated here (not imported) for the same
    reason `run_segmenter_inspect.py` duplicates rather than imports
    `preprocessing_images.py` - see that file's module docstring
    (top-level side effects + placeholder paths at import time).
    """
    cv2mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
    final_mask = np.zeros_like(cv2mask)
    cv2.drawContours(final_mask, circular_contours, -1, (255,), thickness=cv2.FILLED)
    return (final_mask / 255.0).astype(float)


# ---------------------------------------------------------------------------
# --mode pipeline: named config runners.
# ---------------------------------------------------------------------------

def run_config_a(model, img, min_circularity=0.7):
    """
    Config A: faithful reproduction of preprocessing_images.py's
    process_image() (production today). Segment the RAW image first; keep
    the circularity-union filtered mask; if empty, retry ONCE on a
    brightness/contrast-corrected copy; if STILL empty, fall back to the
    full frame. `raw_mask` in the returned dict is whichever forward
    pass's output actually ended up feeding the (possibly-empty) filtered
    mask - i.e. the retry's output if a retry happened, else the first
    pass's. Crop style is "warp": production crops directly to this
    (possibly non-square) box and force-resizes to 224x224, which distorts
    the embryo's shape whenever the box isn't square - see
    `compute_crop_diagnostics`.
    """
    raw_mask = segment_image(model, img)
    filtered_mask = filter_mask_by_circularity_union(raw_mask, min_circularity)
    brightness_fallback = False
    if np.max(filtered_mask) == 0:
        brightness_fallback = True
        bright_img = automatic_brightness_and_contrast(img)
        raw_mask = segment_image(model, bright_img)
        filtered_mask = filter_mask_by_circularity_union(raw_mask, min_circularity)
    full_image_fallback = False
    if np.max(filtered_mask) == 0:
        full_image_fallback = True
        filtered_mask = np.ones_like(filtered_mask)
    bbox = find_bounding_box(filtered_mask)
    return {
        "raw_mask": raw_mask, "filtered_mask": filtered_mask, "bbox": bbox,
        "crop_style": "warp",
        "brightness_fallback_triggered": brightness_fallback,
        "full_image_fallback_triggered": full_image_fallback,
    }


def run_config_new_pipeline(model, img, crop_style, max_area_frac=0.9, border_margin=2,
                             min_area_frac=0.1):
    """
    The "Phase 1" pipeline from run_segmenter_inspect.py, parameterized by
    `model` (production or SegmenterModelSkip weights) and `crop_style`
    ("warp" or "squarify") to cover configs B', B, and C with one function:
      Bprime = run_config_new_pipeline(prod_model, img, "warp", ...)
      B      = run_config_new_pipeline(prod_model, img, "squarify", ...)
      C      = run_config_new_pipeline(skip_model, img, "squarify", ...)

    ALWAYS brightness-corrects first (unlike config A's conditional
    retry - there is only ever one forward pass here), then picks a
    single best contour via `select_best_contour` (no circularity check).
    `crop_style` affects ONLY how `compute_crop_diagnostics` reports
    distortion below - it does not change mask/bbox selection itself,
    since squarify_bbox() is a post-hoc padding step applied to whatever
    bbox `find_bounding_box` returns either way. This makes B/Bprime
    directly comparable on mask/bbox metrics (isolating the crop-style
    effect to just the distortion diagnostic), while B/C differ on
    everything (different weights).
    """
    bright_img = automatic_brightness_and_contrast(img)
    raw_mask = segment_image(model, bright_img)
    filtered_mask, _selected_contour = select_best_contour(
        raw_mask, max_area_frac, border_margin, min_area_frac)
    full_image_fallback = False
    if np.max(filtered_mask) == 0:
        full_image_fallback = True
        filtered_mask = np.ones_like(filtered_mask)
    bbox = find_bounding_box(filtered_mask)
    return {
        "raw_mask": raw_mask, "filtered_mask": filtered_mask, "bbox": bbox,
        "crop_style": crop_style,
        "brightness_fallback_triggered": None,  # N/A - always brightness-corrected, no retry concept
        "full_image_fallback_triggered": full_image_fallback,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_image_mask(image_path, mask_path, size=(224, 224)):
    """--mode model's loader: mirrors SegmentationDataset in
    train_segmenter.py (minus augmentation). Returns the image NORMALIZED
    to [0, 1] (as `predict_prob` expects), unlike `load_image` above."""
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
    """--mode pipeline's ground-truth-mask loader: nearest-neighbor resize
    (to stay binary) + threshold at 127, same convention as
    `load_image_mask` above and `labelme_json_to_mask.py`. Kept separate
    from `load_image_mask` because pipeline mode loads the raw image via
    `load_image` (unnormalized, matching segment_image()'s expectations)
    rather than the [0, 1]-normalized array `load_image_mask` produces."""
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.float32) > 127


def predict_prob(model, img_arr, device="cpu"):
    """--mode model's forward pass, returning the raw (pre-threshold)
    224x224 probability map - same forward pass as `segment_image()`,
    just without the threshold applied, since the threshold-sweep and
    calibration-curve metrics need the full probability surface."""
    x = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.detach().cpu().numpy()[0, 0]


def collect_labeled_pairs_with_dataset_tags(sources):
    """
    --mode pipeline's pair collector. `sources` is a list of
    (dataset_name, images_dir, masks_dir, mask_suffix, mask_ext) tuples.
    Returns a flat list of (dataset_name, stem, image_path, mask_path) -
    ALL matched pairs from ALL sources, each tagged with its source. This
    deliberately does NOT merge/override by stem the way
    `train_segmenter.discover_pairs` does (that behavior is right for
    TRAINING - "replace old inaccurate masks with corrected ones" - but
    wrong for evaluation, where we want every source's images visible and
    separately reportable, even if two sources happened to share a stem).
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
    """Identical formula to `dice_iou_at_threshold()` in
    train_segmenter.py (same epsilon), just operating on a single
    already-binarized numpy mask pair instead of a torch batch."""
    pred_bin = pred_bin.astype(np.float64).ravel()
    gt_bin = gt_bin.astype(np.float64).ravel()
    intersection = float((pred_bin * gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    union = denom - intersection
    dice = (2 * intersection + eps) / (denom + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def _border_mask(mask_bool):
    """3x3-connectivity boundary (outer ring) of a binary mask, via
    erosion. Uses cv2 (already a hard dependency of this repo) rather
    than adding scipy just for `binary_erosion`/`distance_transform_edt`."""
    if not mask_bool.any():
        return mask_bool.copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    border = cv2.bitwise_and(mask_u8, cv2.bitwise_not(eroded))
    return border.astype(bool)


def _distance_map_to(border_bool):
    """Per-pixel Euclidean distance to the nearest True pixel of
    `border_bool`, via cv2.distanceTransform (which measures distance to
    the nearest ZERO pixel - so border pixels are set to 0, everything
    else to 255, before calling it)."""
    src = np.where(border_bool, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def hausdorff_distances(pred_bin, gt_bin, percentile=95.0):
    """
    Returns (hd, hd_p): the full (100th-percentile/max) Hausdorff distance
    and the `percentile`-th percentile Hausdorff distance (HD95 when
    percentile=95, the default), both in pixels at whatever resolution the
    input masks are (224x224 for this script - see module docstring).

    Computed the standard symmetric way used in medical-image segmentation
    benchmarks (e.g. MONAI/medpy): for every boundary pixel of the
    prediction, find its distance to the nearest boundary pixel of the
    ground truth (and vice versa); HD is the max over the union of both
    directed distance sets, HD95 is the max of the two directed
    95th-percentiles.

    Why HD95 in addition to raw HD: HD is extremely sensitive to a single
    outlier pixel. HD95 discards the worst 5% of boundary distances before
    taking the max, which is far more representative of overall boundary
    quality.

    Returns (None, None) if either mask is entirely empty - Hausdorff
    distance is undefined in that case.
    """
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
    """
    --mode model only. Bins every pixel's predicted probability into
    `n_bins` equal-width bins over [0, 1], and for each bin reports the
    mean predicted probability, the empirical fraction of those pixels
    that are actually embryo, and the pixel count. Also returns ECE
    (Expected Calibration Error) and Brier score - see original docstring
    detail preserved from this script's first version.
    """
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
    """Fraction of ground-truth embryo pixels that fall inside `box` -
    directly answers "did this crop lose part of the embryo?" (a failure
    mode Dice/IoU on the mask alone doesn't cleanly capture, since it's
    about the CROP, not the mask). Returns None if the GT mask is empty
    (shouldn't happen with real annotations, but guarded defensively)."""
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
    Failure-mode diagnostics for a predicted bounding box, mirroring
    run_segmenter_inspect.py's per-image diagnostics:
      - crop_area_fraction: fraction of the frame the box covers.
      - bbox_aspect_ratio: width/height of the RAW box (before any
        squarify padding).
      - touches_border: proxy for under-recall or a genuinely clipped
        source image.
      - possible_background_false_positive: crop > `background_area_flag_
        threshold` of the frame - likely a dish-rim/vignette false
        detection rather than a genuinely embryo-filling frame.
      - likely_visible_distortion: True only for `crop_style="warp"` boxes
        whose aspect ratio deviates >20% from 1.0 (production's actual
        bug); ALWAYS False for `crop_style="squarify"`, since
        `squarify_bbox()` pads to a square BEFORE resizing, making a
        non-uniform final resize impossible by construction, regardless
        of how asymmetric the raw box is. This is what lets configs B/C
        directly demonstrate the crop-distortion fix numerically.
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
    """Builds one --mode pipeline per-image CSV row from a
    run_config_a()/run_config_new_pipeline() result plus the ground-truth
    mask: Tier 1 (raw mask) + Tier 2 (post-contour mask) Dice/IoU/HD95,
    Tier 3 (bbox IoU, embryo recall, centroid distance, crop diagnostics),
    fallback flags, and latency."""
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
    """Mean/rate aggregation over a list of evaluate_pipeline_result()
    rows (e.g. all rows for one config, or one (config, dataset) group)."""
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
    """
    Storage/deployability numbers for a loaded checkpoint:
      - n_params: trainable parameter count
      - n_params_m: same, in millions (for plotting)
      - weights_file_mb: on-disk .pt file size
      - model_ram_mb: in-memory footprint of loaded parameters + buffers
        (actual element sizes, not just n_params * 4)
    """
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


def build_pipeline_resource_rows(summary_all, prod_model, skip_model, prod_weights, skip_weights):
    """One deployability row per pipeline config (ALL-dataset summary only)."""
    storage_prod = model_storage_stats(prod_model, prod_weights) if prod_model else None
    storage_skip = model_storage_stats(skip_model, skip_weights) if skip_model else None
    rows = []
    for summary in summary_all:
        config = summary["config"]
        storage = storage_skip if config == "C" else storage_prod
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


def default_model_name(weights_path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(weights_path)))
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    return f"{parent}/{stem}"


# ---------------------------------------------------------------------------
# Plots: --mode model
# ---------------------------------------------------------------------------

def plot_threshold_sweep(sweep_rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping threshold_sweep.png "
              "(pip install matplotlib to enable; numbers are still in threshold_sweep.csv).")
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
    """
    Headline benchmark figure: grouped bar chart of mean Dice and mean IoU
    per model at the fixed reporting threshold, with +/-1 std error bars
    computed across images.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping benchmark_dice_iou.png "
              "(pip install matplotlib to enable).")
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping per_image_dice_iou.png "
              "(pip install matplotlib to enable).")
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping calibration.png "
              "(pip install matplotlib to enable; numbers are still in calibration_bins.csv).")
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
    """Grouped bar chart: mean Tier-2 (post-contour-filtering) Dice/IoU
    +/- std, one group of bars per pipeline config, combined across all
    dataset sources ('ALL'). Same visual style as plot_benchmark_dice_iou
    (--mode model's equivalent figure), but for full pipelines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_dice_iou.png.")
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
    """Grouped bar chart: bbox IoU vs. GT and embryo recall-in-bbox, one
    group per pipeline config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_bbox_metrics.png.")
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
    """Grouped bar chart of failure-mode/deployability rates (fraction of
    images) per pipeline config: full-image fallback, border-touch,
    possible dish/vignette false positive, visible aspect-ratio
    distortion. Lower is better for all four."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_failure_rates.png.")
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
    """Per-image Tier-2 (post-contour) Dice/IoU distributions per pipeline
    config (box plot + jittered points)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_per_image_dice_iou.png.")
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
    """Grouped bar chart: x = dataset source, groups = pipeline config,
    y = mean metric. Lets you see whether a config's ranking holds up
    consistently across the old paper dataset and the newer Labelme-
    labeled set, rather than trusting one pooled number - important since
    a model may have literally been trained on some of these images (see
    the caution message printed at the end of run_pipeline_mode)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: matplotlib not installed, skipping {os.path.basename(out_path)}.")
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_resources.png.")
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
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Pipeline deployability: time + storage (all datasets combined)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote pipeline time/storage plot to {os.path.abspath(out_path)}")


def plot_pipeline_latency_box(per_image_rows, out_path):
    """Per-image end-to-end latency distribution per pipeline config (box plot)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping pipeline_latency_box.png.")
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


def warmup_pipeline_models(prod_model, skip_model, img_arr, args, n_warmup):
    """Run a few untimed iterations so latency measurements aren't dominated
    by one-time CPU cache / PyTorch thread-pool warmup effects."""
    if n_warmup <= 0:
        return
    if prod_model is not None:
        for _ in range(n_warmup):
            run_config_a(prod_model, img_arr, min_circularity=args.circularity_threshold)
            run_config_new_pipeline(prod_model, img_arr, "squarify", args.max_area_frac,
                                    args.border_margin, args.min_area_frac)
    if skip_model is not None:
        for _ in range(n_warmup):
            run_config_new_pipeline(skip_model, img_arr, "squarify", args.max_area_frac,
                                    args.border_margin, args.min_area_frac)


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
        dataset_names = _expand_per_pair(args.dataset_name, n_pairs, "dataset-name", None)
        if dataset_names[0] is None:
            dataset_names = [os.path.basename(os.path.normpath(d)) for d in args.images_dir]
        sources = list(zip(dataset_names, args.images_dir, args.masks_dir, mask_suffixes, mask_exts))
    else:
        sources = DEFAULT_PIPELINE_SOURCES
        print("No --images-dir given; using this repo's 3 default labeled sources "
              "(old paper train + old paper val + new Labelme labels):")

    print("Discovering labeled (image, mask) pairs across all dataset sources...")
    pairs = collect_labeled_pairs_with_dataset_tags(sources)
    if not pairs:
        print("ERROR: no (image, mask) pairs found across any source.", file=sys.stderr)
        sys.exit(1)
    print(f"Total labeled pairs across all sources: {len(pairs)}\n")

    configs_to_run = [c.strip() for c in args.configs.split(",") if c.strip()]
    valid_configs = {"A", "Bprime", "B", "C"}
    unknown = set(configs_to_run) - valid_configs
    if unknown:
        print(f"ERROR: unknown --configs entries {sorted(unknown)}; choices are "
              f"{sorted(valid_configs)}.", file=sys.stderr)
        sys.exit(1)

    needs_prod = any(c in ("A", "Bprime", "B") for c in configs_to_run)
    needs_skip = "C" in configs_to_run

    prod_model = skip_model = None
    n_params_prod = n_params_skip = None
    if needs_prod:
        if not os.path.exists(args.prod_weights):
            print(f"ERROR: production weights not found at {args.prod_weights}", file=sys.stderr)
            sys.exit(1)
        prod_model = load_model(args.prod_weights)
        n_params_prod = sum(p.numel() for p in prod_model.parameters())
    if needs_skip:
        if not os.path.exists(args.skip_weights):
            print(f"ERROR: skip-connections weights not found at {args.skip_weights}", file=sys.stderr)
            sys.exit(1)
        skip_model = load_model(args.skip_weights)
        n_params_skip = sum(p.numel() for p in skip_model.parameters())

    print("Running configs (CPU-only, matching production/run_segmenter_inspect.py):")
    for c in configs_to_run:
        print(f"  {c}: {CONFIG_DESCRIPTIONS[c]}")
    print()

    if args.timing_warmup > 0 and pairs:
        _, _, warmup_path, _ = pairs[0]
        warmup_img = load_image(warmup_path)
        print(f"Timing warmup: {args.timing_warmup} untimed iteration(s) per loaded model "
              f"on {os.path.basename(warmup_path)} ...")
        warmup_pipeline_models(prod_model, skip_model, warmup_img, args, args.timing_warmup)
        print()

    per_image_rows = []
    for dataset_name, stem, img_path, mask_path in pairs:
        img_arr = load_image(img_path)
        img_h, img_w = img_arr.shape[0], img_arr.shape[1]
        gt_bin = load_gt_mask(mask_path, size=(img_w, img_h))

        for config in configs_to_run:
            model = skip_model if config == "C" else prod_model
            t0 = time.perf_counter()
            if config == "A":
                result = run_config_a(model, img_arr, min_circularity=args.circularity_threshold)
            elif config == "Bprime":
                result = run_config_new_pipeline(model, img_arr, "warp", args.max_area_frac,
                                                  args.border_margin, args.min_area_frac)
            elif config == "B":
                result = run_config_new_pipeline(model, img_arr, "squarify", args.max_area_frac,
                                                  args.border_margin, args.min_area_frac)
            else:  # "C"
                result = run_config_new_pipeline(model, img_arr, "squarify", args.max_area_frac,
                                                  args.border_margin, args.min_area_frac)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            row = evaluate_pipeline_result(config, dataset_name, stem, result, gt_bin, img_w, img_h,
                                           args.hd_percentile, args.background_area_flag_threshold,
                                           latency_ms)
            per_image_rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "pipeline_per_image_metrics.csv"), per_image_rows)

    n_params_by_config = {c: (n_params_skip if c == "C" else n_params_prod) for c in configs_to_run}
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

    for row in summary_rows:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)

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

    resource_rows = build_pipeline_resource_rows(
        summary_all, prod_model, skip_model, args.prod_weights, args.skip_weights)
    for row in resource_rows:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
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
# Main
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the embryo segmenter. --mode model (default): raw CNN mask "
                    "quality (Dice/IoU/HD95/threshold-sweep/calibration) for one or more "
                    "checkpoints. --mode pipeline: full end-to-end preprocessing-pipeline "
                    "comparison (model + contour-selection + crop) across named configs "
                    "A/Bprime/B/C - see module docstring.")
    parser.add_argument("--mode", choices=["model", "pipeline"], default="model",
                        help="'model' (default) or 'pipeline' - see module docstring for what "
                             "each evaluates.")

    data = parser.add_argument_group("data selection")
    data.add_argument("--images-dir", action="append", default=None,
                      help="Directory of raw images (repeatable). --mode model default: "
                           f"{DEFAULT_IMAGES_DIR}. --mode pipeline default (if omitted): all 3 "
                           "of this repo's labeled sources combined (see module docstring).")
    data.add_argument("--masks-dir", action="append", default=None,
                      help="Directory of mask files matching --images-dir (repeatable). "
                           f"--mode model default: {DEFAULT_MASKS_DIR}.")
    data.add_argument("--mask-suffix", action="append", default=None,
                      help="Mask filename suffix, one per --images-dir pair (default: _mask, "
                           "matching labelme_json_to_mask.py - see train_segmenter.py). Pass "
                           "'' for the original paper dataset (mask named identically to the "
                           "image, e.g. --mask-suffix '' --mask-ext .jpg).")
    data.add_argument("--mask-ext", action="append", default=None,
                      help="Mask file extension, one per --images-dir pair (default: .png).")
    data.add_argument("--dataset-name", action="append", default=None,
                      help="(--mode pipeline only) Display name for each --images-dir/"
                           "--masks-dir pair, for the per-dataset breakdown in "
                           "pipeline_summary.csv/pipeline_dice_by_dataset.png (repeatable). "
                           "Default: derived from each --images-dir's basename.")

    model_grp = parser.add_argument_group("--mode model options")
    model_grp.add_argument("--weights", action="append", default=None,
                           help="Path to a segmenter .pt checkpoint (repeatable, to compare "
                                f"multiple checkpoints in one run). Default: {DEFAULT_WEIGHTS} "
                                "(current production weights). Architecture is auto-detected "
                                "per checkpoint.")
    model_grp.add_argument("--weights-name", action="append", default=None,
                           help="Display name for each --weights entry, in the same order "
                                "(repeatable). Default: derived from each path's "
                                "parent-dir/filename.")
    model_grp.add_argument("--threshold", type=float, default=0.5,
                           help="Fixed reporting threshold for per_image_metrics.csv/"
                                "summary.csv (default: 0.5, matching production's "
                                "segment_image()).")
    model_grp.add_argument("--threshold-min", type=float, default=0.05,
                           help="Lowest threshold in the sweep (default: 0.05).")
    model_grp.add_argument("--threshold-max", type=float, default=0.95,
                           help="Highest threshold in the sweep (default: 0.95).")
    model_grp.add_argument("--threshold-step", type=float, default=0.05,
                           help="Step size for the threshold sweep (default: 0.05).")
    model_grp.add_argument("--calibration-bins", type=int, default=10,
                           help="Number of probability bins for the reliability "
                                "diagram/ECE (default: 10).")

    pipeline_grp = parser.add_argument_group("--mode pipeline options")
    pipeline_grp.add_argument("--prod-weights", default=DEFAULT_WEIGHTS,
                              help="Production (non-skip-connection) checkpoint, used by "
                                   f"configs A/Bprime/B (default: {DEFAULT_WEIGHTS}).")
    pipeline_grp.add_argument("--skip-weights", default=DEFAULT_SKIP_WEIGHTS,
                              help="SegmenterModelSkip checkpoint, used by config C "
                                   f"(default: {DEFAULT_SKIP_WEIGHTS}).")
    pipeline_grp.add_argument("--configs", default="A,Bprime,B,C",
                              help="Comma-separated subset of pipeline configs to run "
                                   "(default: all four - A,Bprime,B,C). A=production today; "
                                   "Bprime=new contour logic + old warp crop (ablation); "
                                   "B=new pipeline, old model; C=new pipeline, new model. "
                                   "See module docstring for full descriptions.")
    pipeline_grp.add_argument("--max-area-frac", type=float, default=0.9,
                              help="select_best_contour's implausibly-large-contour reject "
                                   "threshold, as a fraction of the frame (default: 0.9). "
                                   "Used by configs Bprime/B/C.")
    pipeline_grp.add_argument("--border-margin", type=int, default=2,
                              help="select_best_contour's border-touch pixel margin "
                                   "(default: 2). Used by configs Bprime/B/C.")
    pipeline_grp.add_argument("--min-area-frac", type=float, default=0.1,
                              help="select_best_contour's noise-speck-rejection area floor, "
                                   "as a fraction of the frame (default: 0.1, i.e. 10%%). "
                                   "Used by configs Bprime/B/C. This now matches "
                                   "select_best_contour()'s own default and the "
                                   "run_segmenter_inspect.py CLI default everywhere (a "
                                   "previous mismatch where that CLI default was 0.01 despite "
                                   "its help text already claiming 0.1 has been fixed).")
    pipeline_grp.add_argument("--circularity-threshold", type=float, default=0.7,
                              help="Config A's circularity-union filter threshold, matching "
                                   "preprocessing_images.py's hardcoded value (default: 0.7).")
    pipeline_grp.add_argument("--background-area-flag-threshold", type=float, default=0.7,
                              help="Flag a crop as 'possible_background_false_positive' if it "
                                   "covers more than this fraction of the frame (default: "
                                   "0.7). Matches run_segmenter_inspect.py's CLI default.")
    pipeline_grp.add_argument("--timing-warmup", type=int, default=3,
                              help="Untimed pipeline iterations per loaded model before "
                                   "measuring latency (default: 3). Set to 0 to disable. "
                                   "Reduces one-time CPU/PyTorch warmup bias in timing stats.")

    shared = parser.add_argument_group("shared options")
    shared.add_argument("--hd-percentile", type=float, default=95.0,
                        help="Percentile for the 'HD95' metric (default: 95.0). Set to 100 "
                             "to make hd95 identical to the raw max Hausdorff distance.")
    shared.add_argument("--output-dir", default=None,
                        help="Directory to write CSVs/plots to. Default: "
                             "./segmenter_eval_output (--mode model) or "
                             "./segmenter_pipeline_eval_output (--mode pipeline).")
    shared.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto-detect). --mode model only - "
                             "--mode pipeline is always CPU-only, matching production/"
                             "run_segmenter_inspect.py (segment_image() doesn't accept a "
                             "device argument).")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "segmenter_eval_output" if args.mode == "model" else "segmenter_pipeline_eval_output"

    if args.mode == "model":
        run_model_mode(args)
    else:
        run_pipeline_mode(args)


if __name__ == "__main__":
    main()
