"""
eval_segmenter.py

Quantitative benchmarking for `EmbryoSegmenter/SegmenterModel.py` (and its
`SegmenterModelSkip` U-Net variant from `train_segmenter.py`) against a set
of ground-truth masks. Fills the gap noted while discussing benchmarking
approaches for this repo: `train_segmenter.py` reports Dice/IoU as a
side-effect of training, but there was no standalone "just evaluate a
checkpoint" tool, and Hausdorff/HD95 and threshold-calibration analysis
didn't exist anywhere.

Reuses this repo's existing conventions rather than reinventing them:
  - Pair discovery: `discover_pairs()` from `train_segmenter.py` (same
    filename-stem matching, same `--mask-suffix`/`--mask-ext` semantics).
  - Model loading: `load_model()` from `run_segmenter_inspect.py` (same
    auto-detection of plain `SegmenterModel` vs. skip-connection
    `SegmenterModelSkip` checkpoints, from tensor shapes - no flag needed).
  - Image/mask resizing: same as `SegmentationDataset` in
    `train_segmenter.py` (image: bilinear, mask: nearest, resized together
    to the model's fixed 224x224 input/output size). Consequently, all
    pixel-distance-based metrics below (Hausdorff/HD95) are reported in
    224x224-pixel units, NOT the original annotated image's resolution -
    same scale Dice/IoU are already implicitly computed at elsewhere in
    this repo.
  - Dice/IoU formula: identical to `dice_iou_at_threshold()` in
    `train_segmenter.py` (same epsilon, same definition), just evaluated
    per-image here instead of per-batch during training.

Default data source: `preprocessing/new_labels/raw_images` +
`preprocessing/new_labels/masks` (this repo's current hand-labeled set,
named `<stem>.jpg` + `<stem>_mask.png` - the `labelme_json_to_mask.py`
convention). Point `--images-dir`/`--masks-dir` elsewhere to evaluate on a
different labeled set instead.

What this evaluates: the raw CNN probability mask (`segment_image()`'s
output before thresholding), i.e. pure segmentation quality - NOT the
downstream contour-selection/bounding-box/crop pipeline in
`preprocessing_images.py` or `run_segmenter_inspect.py`. This matches what
`train_segmenter.py`'s own Dice/IoU already measure, and is the right
level for "is the mask itself accurate" questions; see
`run_segmenter_inspect.py` (visual crop inspection) and
`SEGMENTER_ACCURACY_ROADMAP.md` (contour-selection failure modes) for the
downstream-pipeline-level questions instead.

Metrics computed:
  1. Dice + IoU at a fixed reporting threshold (--threshold, default 0.5 -
     matches production's `segment_image()`).
  2. Hausdorff distance (HD) AND HD95 (the 95th-percentile Hausdorff
     distance - see `hausdorff_distances()` docstring for why HD95 is
     generally preferred over raw HD: a single stray/mislabeled boundary
     pixel can otherwise dominate the metric).
  3. A threshold sweep (--threshold-min/--threshold-max/--threshold-step)
     reporting mean Dice/IoU/HD95 at each threshold, so you can see
     whether 0.5 is actually the best operating point for a given
     checkpoint (production hardcodes 0.5; a fine-tuned/skip-connection
     checkpoint may calibrate differently).
  4. A pixel-probability reliability diagram (calibration curve) plus two
     scalar calibration summaries - Expected Calibration Error (ECE) and
     Brier score - computed once per model across ALL pixels of ALL
     images (this is threshold-INDEPENDENT, unlike #1-3: it asks "when
     the model says 0.7, is the pixel actually embryo ~70% of the time?",
     not "what's the best cutoff").

Supports comparing multiple checkpoints side by side in one run (e.g. the
production weights vs. a `--skip-connections` checkpoint from
`train_segmenter.py`) via repeatable --weights/--weights-name.

Usage:
    cd preprocessing
    # Evaluate production weights on the default new_labels/ set:
    python eval_segmenter.py

    # Compare production vs. a fine-tuned/skip-connection checkpoint:
    python eval_segmenter.py \\
        --weights EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt --weights-name production \\
        --weights EmbryoSegmenter/weights_skip/latest.pt --weights-name skip_connections

    # Evaluate on a different labeled set:
    python eval_segmenter.py --images-dir path/to/raw --masks-dir path/to/masks

Outputs (written to --output-dir, default ./segmenter_eval_output):
    per_image_metrics.csv   one row per (model, image): dice, iou, hd, hd95 @ --threshold
    threshold_sweep.csv     one row per (model, threshold): mean dice/iou/hd95
    calibration_bins.csv    one row per (model, probability bin): mean_pred, empirical_frac, count
    summary.csv             one row per model: headline numbers @ --threshold, best threshold, ECE, Brier
    threshold_sweep.png     Dice/IoU/HD95 vs. threshold, one line per model (requires matplotlib)
    calibration.png         reliability diagram, one curve per model (requires matplotlib)
    benchmark_dice_iou.png  headline benchmark figure: mean Dice/IoU ± std per model (bar chart)
    per_image_dice_iou.png  per-image Dice/IoU distributions per model (box plot)

No new dependencies: only numpy/torch/opencv/pillow (already required by
EmbryoSegmenter/requirements-segmenter.txt) plus matplotlib for the two
plots (also already listed there, for train_segmenter.py's curves; if
it's missing, the CSVs are still written and only the plots are skipped).
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from run_segmenter_inspect import DEFAULT_WEIGHTS, load_model
from train_segmenter import discover_pairs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "new_labels", "raw_images")
DEFAULT_MASKS_DIR = os.path.join(SCRIPT_DIR, "new_labels", "masks")


# ---------------------------------------------------------------------------
# Data loading (mirrors SegmentationDataset in train_segmenter.py, minus
# augmentation - this is evaluation, not training).
# ---------------------------------------------------------------------------

def load_image_mask(image_path, mask_path, size=(224, 224)):
    img = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)
    img = img.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return img_arr, mask_arr


def predict_prob(model, img_arr, device="cpu"):
    """One forward pass, returning the raw (pre-threshold) 224x224
    probability map as a numpy array - same forward pass as
    `segment_image()` in run_segmenter_inspect.py, just without the
    threshold applied, since the metrics below need the full probability
    surface (for the threshold sweep and calibration curve)."""
    x = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.detach().cpu().numpy()[0, 0]


# ---------------------------------------------------------------------------
# Metrics
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
    the nearest ZERO pixel - so the border pixels are set to 0 and
    everything else to 255 before calling it)."""
    src = np.where(border_bool, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def hausdorff_distances(pred_bin, gt_bin, percentile=95.0):
    """
    Returns (hd, hd_p): the full (100th-percentile / max) Hausdorff
    distance and the `percentile`-th percentile Hausdorff distance
    (HD95 when percentile=95, the default), both in pixels at whatever
    resolution the input masks are (224x224 for this script - see module
    docstring).

    Computed the standard symmetric way used in medical-image
    segmentation benchmarks (e.g. MONAI/medpy): for every boundary pixel
    of the prediction, find its distance to the nearest boundary pixel
    of the ground truth (and vice versa); HD is the max over the union of
    both directed distance sets, HD95 is the max of the two directed
    95th-percentiles. Using the boundary/surface pixels (not all
    foreground pixels) matches the standard definition and is much
    cheaper than an all-pixels comparison.

    Why HD95 in addition to raw HD: HD is extremely sensitive to a
    single outlier pixel (e.g. one stray thresholding artifact at the
    mask edge can make an otherwise-excellent mask look terrible). HD95
    discards the worst 5% of boundary distances before taking the max,
    which is far more representative of overall boundary quality - this
    is why HD95 (not raw HD) is the standard reported metric in most
    segmentation papers.

    Returns (None, None) if either mask is entirely empty (no foreground
    pixels at all) - Hausdorff distance is undefined in that case, since
    there is no boundary to measure to/from. This can legitimately happen
    at low thresholds (empty prediction) or, in principle, for a
    genuinely blank ground-truth mask.
    """
    if not pred_bin.any() or not gt_bin.any():
        return None, None

    pred_border = _border_mask(pred_bin.astype(bool))
    gt_border = _border_mask(gt_bin.astype(bool))
    # Degenerate case (e.g. a mask covering the entire frame, so erosion
    # leaves no "outer ring" against the image edge): fall back to
    # treating every foreground pixel as its own boundary rather than
    # silently returning an empty/undefined result.
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
    Bins every pixel's predicted probability (across all evaluated
    images) into `n_bins` equal-width bins over [0, 1], and for each bin
    reports the mean predicted probability, the empirical fraction of
    those pixels that are actually embryo (ground truth), and the pixel
    count. A well-calibrated model has mean_pred ~= empirical_frac in
    every bin (points on the y=x diagonal in the resulting plot).

    Also returns two scalar summaries:
      - ECE (Expected Calibration Error): the pixel-count-weighted mean
        absolute gap between predicted and empirical probability across
        bins. Lower is better; 0 is perfect calibration.
      - Brier score: mean squared error between predicted probability
        and the true 0/1 label, pooled over all pixels. Lower is better;
        it also rewards sharpness (confident, correct predictions), not
        just calibration, so read it alongside ECE rather than alone.
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
    per model at the fixed reporting threshold, with ±1 std error bars
    computed across images. This is the standard way segmentation papers
    report a model comparison in a figure (the numeric table lives in
    summary.csv; this is the visual counterpart).
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
    """
    Per-image Dice/IoU distributions per model (box plot + jittered points).
    Useful for benchmarking write-ups to show not just the mean but spread
    and outliers across the evaluated set.
    """
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SegmenterModel/SegmenterModelSkip checkpoints: Dice, IoU, "
                    "Hausdorff distance + HD95, a threshold sweep, and pixel-probability "
                    "calibration, against ground-truth masks.")
    parser.add_argument("--images-dir", action="append", default=None,
                        help="Directory of raw images (repeatable, same convention as "
                             f"train_segmenter.py). Default: {DEFAULT_IMAGES_DIR}")
    parser.add_argument("--masks-dir", action="append", default=None,
                        help="Directory of mask PNGs matching --images-dir (repeatable). "
                             f"Default: {DEFAULT_MASKS_DIR}")
    parser.add_argument("--mask-suffix", action="append", default=None,
                        help="Mask filename suffix, one per --images-dir pair (default: "
                             "_mask, matching labelme_json_to_mask.py - see train_segmenter.py).")
    parser.add_argument("--mask-ext", action="append", default=None,
                        help="Mask file extension, one per --images-dir pair (default: .png).")
    parser.add_argument("--weights", action="append", default=None,
                        help="Path to a segmenter .pt checkpoint (repeatable, to compare "
                             f"multiple checkpoints in one run). Default: {DEFAULT_WEIGHTS} "
                             "(current production weights). Architecture (plain SegmenterModel "
                             "vs. skip-connection SegmenterModelSkip) is auto-detected per "
                             "checkpoint, same as run_segmenter_inspect.py.")
    parser.add_argument("--weights-name", action="append", default=None,
                        help="Display name for each --weights entry, in the same order "
                             "(repeatable). Default: derived from each path's parent-dir/filename.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fixed reporting threshold for per_image_metrics.csv/summary.csv "
                             "(default: 0.5, matching production's segment_image()). Always "
                             "included as one of the threshold-sweep points too.")
    parser.add_argument("--threshold-min", type=float, default=0.05,
                        help="Lowest threshold in the sweep (default: 0.05).")
    parser.add_argument("--threshold-max", type=float, default=0.95,
                        help="Highest threshold in the sweep (default: 0.95).")
    parser.add_argument("--threshold-step", type=float, default=0.05,
                        help="Step size for the threshold sweep (default: 0.05).")
    parser.add_argument("--hd-percentile", type=float, default=95.0,
                        help="Percentile for the 'HD95' metric (default: 95.0). Set to 100 "
                             "to make hd95 identical to the raw max Hausdorff distance.")
    parser.add_argument("--calibration-bins", type=int, default=10,
                        help="Number of probability bins for the reliability diagram/ECE "
                             "(default: 10).")
    parser.add_argument("--output-dir", default="segmenter_eval_output",
                        help="Directory to write CSVs/plots to (default: ./segmenter_eval_output).")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect).")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
