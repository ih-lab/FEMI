"""
train_segmenter.py

Train or fine-tune the embryo segmenter on (raw image, binary mask) pairs.
Masks are 0/255 images named "<image_stem><mask_suffix><mask_ext>" (default
"_mask.png", as written by labelme_json_to_mask.py). Image and mask are
resized together to 224x224 at load time.

Data:
  - Repeat --images-dir/--masks-dir for several sources; on a filename stem
    collision the later source wins, so list corrected masks last.
  - The original paper dataset names masks like the image: use
    --mask-suffix "" --mask-ext .jpg.
  - Validation is a random --val-frac split (default 0.2), or a fixed set via
    --val-images-dir/--val-masks-dir, optionally topped up with --extra-val-frac.

Options worth knowing:
  --skip-connections    train SegmenterModelSkip; most production weights still
                        transfer (see load_compatible_state_dict)
  --brightness-correct  apply automatic_brightness_and_contrast() as at inference
  --freeze-encoder      fine-tune only the decoder

Each run writes checkpoints, latest.pt, training_history.csv, and (unless
--no-plot) training_curves.png to --output-dir.

Usage:
    # Check pairing/alignment first:
    python train_segmenter.py --images-dir raw/ --masks-dir masks/ \\
        --preview-only --preview-dir preview/

    # Fine-tune from production weights:
    python train_segmenter.py --images-dir raw/ --masks-dir masks/ \\
        --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned

    # U-Net variant:
    python train_segmenter.py --images-dir raw/ --masks-dir masks/ \\
        --skip-connections --epochs 50 --output-dir EmbryoSegmenter/weights_skip

    # Original paper data + new Labelme labels, with fixed validation sets:
    python train_segmenter.py \\
        --images-dir mask/psd --masks-dir mask/Embryo \\
        --images-dir new_labels/raw_images --masks-dir new_labels/masks \\
        --mask-suffix "" --mask-suffix _mask \\
        --mask-ext .jpg --mask-ext .png \\
        --val-images-dir mask_valid/psd --val-masks-dir mask_valid/Embryo \\
        --val-images-dir new_valid/raw_images --val-masks-dir new_valid/mask_valid \\
        --val-mask-suffix "" --val-mask-suffix _mask \\
        --val-mask-ext .jpg --val-mask-ext .png \\
        --freeze-encoder \\
        --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned

Check a new checkpoint with run_segmenter_inspect.py --weights or
eval_segmenter.py before promoting it.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from EmbryoSegmenter.SegmenterModel import SegmenterModel, SegmenterModelSkip

DEFAULT_INIT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "EmbryoSegmenter", "weights", "epoch_102_loss_0.098710.pt",
)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_compatible_state_dict(model, ckpt_path, device):
    """
    Load only the tensors whose name and shape match `model`; the rest keep
    their random init. From a SegmenterModel checkpoint into
    SegmenterModelSkip, only the conv weights of deconv4-1 and decoded differ.
    Returns (n_loaded, n_total, skipped_layer_names).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    model_state.update(compatible)
    model.load_state_dict(model_state)
    skipped_layers = sorted({k.split(".")[0] for k in skipped})
    return len(compatible), len(model_state), skipped_layers


# ---------------------------------------------------------------------------
# Pair discovery (match by filename stem; later sources win on collisions)
# ---------------------------------------------------------------------------

def _expand_per_pair(values, n_pairs, flag_name, default):
    """Expand a repeatable CLI value to one entry per pair: None or one value
    is broadcast, n_pairs values are used as-is, extra values keep the last
    n_pairs (with a warning), and too few raise ValueError."""
    if values is None:
        values = [default]
    if len(values) == 1:
        return values * n_pairs
    if len(values) == n_pairs:
        return values
    if len(values) > n_pairs:
        trimmed = values[-n_pairs:]
        print(f"WARNING: --{flag_name} given {len(values)} time(s) for {n_pairs} "
              f"--images-dir/--masks-dir pair(s); using the last {n_pairs} "
              f"in order: {trimmed!r}. Pass once (applies to all pairs) or once "
              f"per pair only — not both inline and again at the end.")
        return trimmed
    raise ValueError(
        f"--{flag_name} given {len(values)} time(s) but there are {n_pairs} "
        f"--images-dir/--masks-dir pair(s). Pass it once (applies to all pairs) "
        f"or once per pair, in the same order."
    )


def discover_pairs(images_dirs, masks_dirs, mask_suffixes=None, mask_exts=None):
    """{stem: (image_path, mask_path)} for masks named
    "<stem><mask_suffix><mask_ext>" (default "_mask", ".png"), per pair."""
    if len(images_dirs) != len(masks_dirs):
        raise ValueError("--images-dir and --masks-dir must be given the same number of times "
                          f"(got {len(images_dirs)} images-dir vs {len(masks_dirs)} masks-dir).")
    n_pairs = len(images_dirs)
    mask_suffixes = _expand_per_pair(mask_suffixes, n_pairs, "mask-suffix", "_mask")
    mask_exts = _expand_per_pair(mask_exts, n_pairs, "mask-ext", ".png")

    pairs = {}  # stem -> (image_path, mask_path), later sources override
    for images_dir, masks_dir, mask_suffix, mask_ext in zip(
            images_dirs, masks_dirs, mask_suffixes, mask_exts):
        mask_files = {f for f in os.listdir(masks_dir)} if os.path.isdir(masks_dir) else set()
        n_found = 0
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            mask_name = f"{stem}{mask_suffix}{mask_ext}"
            if mask_name not in mask_files:
                continue
            pairs[stem] = (os.path.join(images_dir, fname), os.path.join(masks_dir, mask_name))
            n_found += 1
        print(f"  {images_dir} + {masks_dir}: {n_found} matched pairs "
              f"({len(os.listdir(images_dir))} images seen, "
              f"{n_found} had a matching '*{mask_suffix}{mask_ext}')")
    return pairs


def load_manifest(manifest_path):
    pairs = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            pairs[stem] = (row["image_path"], row["mask_path"])
    print(f"  {manifest_path}: {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    (image, mask) pairs resized to 224x224 (image bilinear, mask nearest).

    `brightness_correct` runs automatic_brightness_and_contrast() on the full
    frame first, as at inference. With `augment`, the full-resolution pair gets
    a random crop (`crop_min_scale` to 1.0, standing in for the paper's
    cropping and translation), flips, and 90-degree rotations, and the resized
    image alone gets brightness/contrast jitter (not in the paper).
    """

    def __init__(self, pairs, size=(224, 224), augment=False, crop_min_scale=0.85,
                 brightness_correct=False):
        self.pairs = pairs  # list of (image_path, mask_path)
        self.size = size
        self.augment = augment
        self.crop_min_scale = crop_min_scale
        self.brightness_correct = brightness_correct

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, img, mask):
        w, h = img.size
        scale = random.uniform(self.crop_min_scale, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        max_x, max_y = w - new_w, h - new_h
        x0 = random.randint(0, max_x) if max_x > 0 else 0
        y0 = random.randint(0, max_y) if max_y > 0 else 0
        box = (x0, y0, x0 + new_w, y0 + new_h)
        return img.crop(box), mask.crop(box)

    def _augment_pair(self, img, mask):
        img, mask = self._random_crop(img, mask)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.choice([0, 90, 180, 270])
        if k:
            img = img.rotate(k, expand=False)
            mask = mask.rotate(k, expand=False)
        return img, mask

    def _jitter_brightness_contrast(self, img):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        return img

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        img = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        if self.brightness_correct:
            # Local import: eval_segmenter imports this module.
            from eval_segmenter import automatic_brightness_and_contrast
            img = Image.fromarray(automatic_brightness_and_contrast(np.asarray(img)))

        if self.augment:
            img, mask = self._augment_pair(img, mask)

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        if self.augment:
            img = self._jitter_brightness_contrast(img)

        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        img_t = torch.from_numpy(img_arr).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
        return img_t, mask_t


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid outputs; less sensitive to class imbalance than BCE."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


def make_loss(name):
    bce = nn.BCELoss()
    dice = DiceLoss()
    if name == "bce":
        return lambda p, t: bce(p, t)
    if name == "dice":
        return lambda p, t: dice(p, t)
    if name == "bce_dice":
        return lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)
    raise ValueError(f"Unknown loss {name!r}")


@torch.no_grad()
def dice_iou_at_threshold(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    pred_bin = pred_bin.reshape(pred_bin.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred_bin * target).sum(dim=1)
    union = pred_bin.sum(dim=1) + target.sum(dim=1) - intersection
    dice = (2 * intersection + eps) / (pred_bin.sum(dim=1) + target.sum(dim=1) + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            dice, iou = dice_iou_at_threshold(preds.detach(), masks)
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            n_batches += 1
    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def plot_training_curves(history, out_path):
    """Save train vs. val loss/Dice/IoU per epoch. Val loss flattening or rising
    while train loss falls means overfitting. Skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: matplotlib not installed, skipping training-curve plot "
              f"(pip install matplotlib to enable; see EmbryoSegmenter/requirements-segmenter.txt). "
              f"Raw per-epoch numbers are still in "
              f"{os.path.join(os.path.dirname(out_path), 'training_history.csv')}.")
        return

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ("loss", "dice", "iou"), ("Loss", "Dice", "IoU")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], marker=".", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], marker=".", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote training-curve plot to {os.path.abspath(out_path)} - use it to check for "
          f"overfitting (val loss flattening/rising while train loss keeps falling) before "
          f"picking --epochs for your next run.")


def save_preview(dataset, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, len(dataset))
    idxs = random.sample(range(len(dataset)), n)
    for i in idxs:
        img_t, mask_t = dataset[i]
        stem = os.path.splitext(os.path.basename(dataset.pairs[i][0]))[0]
        img = (img_t.squeeze(0).numpy() * 255).astype(np.uint8)
        mask = (mask_t.squeeze(0).numpy() * 255).astype(np.uint8)
        overlay = np.stack([img, np.maximum(img, mask), img], axis=-1)  # green tint where mask=1
        side_by_side = np.concatenate([np.stack([img] * 3, axis=-1), overlay], axis=1)
        Image.fromarray(side_by_side).save(os.path.join(out_dir, f"{stem}_preview.png"))
    print(f"Wrote {n} preview image|mask-overlay pairs to {os.path.abspath(out_dir)} "
          f"- check these BEFORE training that image/mask pairs are correctly aligned "
          f"(not flipped, not off-by-one-stem, correct region highlighted).")


def main():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune the embryo segmenter on (raw image, binary mask) pairs.")
    parser.add_argument("--images-dir", action="append", required=True,
                        help="Raw image directory (repeatable, paired with --masks-dir). "
                             "Later pairs override earlier ones on stem collisions.")
    parser.add_argument("--masks-dir", action="append", required=True,
                        help="Mask directory for each --images-dir.")
    parser.add_argument("--mask-suffix", action="append", default=None,
                        help="Mask filename suffix, once or once per pair (default: _mask; "
                             "'' for the original paper dataset).")
    parser.add_argument("--mask-ext", action="append", default=None,
                        help="Mask file extension, once or once per pair (default: .png; "
                             ".jpg for the original paper dataset).")
    parser.add_argument("--val-mask-suffix", action="append", default=None,
                        help="Like --mask-suffix, for --val-images-dir (default: '').")
    parser.add_argument("--val-mask-ext", action="append", default=None,
                        help="Like --mask-ext, for --val-images-dir (default: .jpg).")
    parser.add_argument("--manifest", default=None,
                        help="Extra CSV of image_path,mask_path pairs, applied last.")
    parser.add_argument("--val-images-dir", action="append", default=[],
                        help="Fixed validation image directory (repeatable); replaces --val-frac.")
    parser.add_argument("--val-masks-dir", action="append", default=[],
                        help="Mask directory matching --val-images-dir, in order.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Random validation fraction (default: 0.2, the paper's 80/20 "
                             "split). Ignored with --val-images-dir.")
    parser.add_argument("--extra-val-frac", type=float, default=0.0,
                        help="With --val-images-dir, also move this fraction of the training "
                             "pool into validation (default: 0; split depends on --seed).")
    parser.add_argument("--crop-min-scale", type=float, default=0.85,
                        help="Smallest random crop as a fraction of the frame (default: 0.85; "
                             "1.0 disables cropping).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce_dice",
                        help="Loss function (default: bce_dice, a 50/50 mix).")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help=f"Checkpoint to fine-tune from (default: {DEFAULT_INIT_WEIGHTS}).")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore --init-weights and start from random initialization.")
    parser.add_argument("--skip-connections", action="store_true",
                        help="Train the U-Net variant SegmenterModelSkip; compatible "
                             "--init-weights tensors still load.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze conv1-conv6 and fine-tune only the decoder (good for small data).")
    parser.add_argument("--brightness-correct", action="store_true",
                        help="Apply the inference brightness/contrast correction to every "
                             "image (requires opencv).")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable train-time augmentation (flips/rotation/brightness jitter).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip training_curves.png.")
    parser.add_argument("--output-dir", default="EmbryoSegmenter/weights_finetuned",
                        help="Checkpoint directory (default: EmbryoSegmenter/weights_finetuned).")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-only", action="store_true",
                        help="Only write image/mask preview overlays, then exit.")
    parser.add_argument("--preview-dir", default="preview",
                        help="Preview output directory (default: preview).")
    parser.add_argument("--preview-n", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Discovering (image, mask) pairs...")
    pairs_by_stem = discover_pairs(args.images_dir, args.masks_dir, args.mask_suffix, args.mask_ext)
    if args.manifest:
        pairs_by_stem.update(load_manifest(args.manifest))
    if not pairs_by_stem:
        print("ERROR: no (image, mask) pairs found. Check --images-dir/--masks-dir/--mask-suffix/--mask-ext.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Total unique pairs after merging all sources: {len(pairs_by_stem)}")

    if args.val_images_dir:
        print("Discovering explicit validation (image, mask) pairs (--val-images-dir given, "
              "--val-frac ignored)...")
        val_mask_suffixes = args.val_mask_suffix if args.val_mask_suffix is not None else [""]
        val_mask_exts = args.val_mask_ext if args.val_mask_ext is not None else [".jpg"]
        val_pairs_by_stem = discover_pairs(args.val_images_dir, args.val_masks_dir,
                                            val_mask_suffixes, val_mask_exts)
        if not val_pairs_by_stem:
            print("ERROR: --val-images-dir given but no (image, mask) pairs found there. "
                  "Check --val-images-dir/--val-masks-dir/--mask-suffix/--mask-ext.",
                  file=sys.stderr)
            sys.exit(1)
        overlap = set(pairs_by_stem) & set(val_pairs_by_stem)
        if overlap:
            print(f"WARNING: {len(overlap)} filename stem(s) appear in BOTH the training and "
                  f"validation sources (e.g. {sorted(overlap)[:3]}) - removing them from the "
                  f"training set to avoid val-on-train leakage.")
            for s in overlap:
                pairs_by_stem.pop(s, None)
        if args.extra_val_frac > 0:
            stems = sorted(pairs_by_stem.keys())
            random.shuffle(stems)
            n_extra = max(1, int(len(stems) * args.extra_val_frac)) if len(stems) > 1 else 0
            extra_val_stems = set(stems[:n_extra])
            train_pairs = [pairs_by_stem[s] for s in stems if s not in extra_val_stems]
            extra_val_pairs = [pairs_by_stem[s] for s in stems if s in extra_val_stems]
            val_pairs = list(val_pairs_by_stem.values()) + extra_val_pairs
            print(f"--extra-val-frac {args.extra_val_frac}: moved {len(extra_val_pairs)} pair(s) "
                  f"from the training pool into validation, on top of the "
                  f"{len(val_pairs_by_stem)}-pair --val-images-dir set - validation now has "
                  f"{len(val_pairs)} pairs total instead of just {len(val_pairs_by_stem)} "
                  f"(a bigger, less noisy validation set, at the cost of {len(extra_val_pairs)} "
                  f"fewer training examples).")
        else:
            train_pairs = list(pairs_by_stem.values())
            val_pairs = list(val_pairs_by_stem.values())
    else:
        stems = sorted(pairs_by_stem.keys())
        random.shuffle(stems)
        n_val = max(1, int(len(stems) * args.val_frac)) if len(stems) > 1 else 0
        val_stems, train_stems = set(stems[:n_val]), stems[n_val:]
        train_pairs = [pairs_by_stem[s] for s in train_stems]
        val_pairs = [pairs_by_stem[s] for s in val_stems]
    print(f"Split: {len(train_pairs)} train / {len(val_pairs)} val")

    if len(train_pairs) < 20:
        print(f"WARNING: only {len(train_pairs)} training examples. With this little data, "
              f"consider --freeze-encoder, heavier reliance on --init-weights fine-tuning "
              f"(don't use --from-scratch), and treating val metrics with caution (high "
              f"variance with a tiny val set).")

    if args.preview_only:
        preview_dataset = SegmentationDataset(train_pairs + val_pairs, augment=False,
                                              brightness_correct=args.brightness_correct)
        save_preview(preview_dataset, args.preview_dir, n=args.preview_n)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SegmentationDataset(train_pairs, augment=not args.no_augment,
                                    crop_min_scale=args.crop_min_scale,
                                    brightness_correct=args.brightness_correct)
    val_ds = SegmentationDataset(val_pairs, augment=False,
                                  brightness_correct=args.brightness_correct)
    print(f"Brightness/contrast correction on train/val images: "
          f"{'ON' if args.brightness_correct else 'OFF'}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if val_pairs else None

    model_cls = SegmenterModelSkip if args.skip_connections else SegmenterModel
    model = model_cls().to(device)
    if args.skip_connections:
        print("Using SegmenterModelSkip (--skip-connections): U-Net variant with skip "
              "connections.")

    if not args.from_scratch:
        if not os.path.exists(args.init_weights):
            print(f"ERROR: --init-weights not found at {args.init_weights}. "
                  f"Pass --from-scratch to train from random initialization instead.",
                  file=sys.stderr)
            sys.exit(1)
        if args.skip_connections:
            n_loaded, n_total, skipped_layers = load_compatible_state_dict(
                model, args.init_weights, device)
            print(f"Initialized {n_loaded}/{n_total} tensors from {args.init_weights} "
                  f"(shape-compatible layers only). Layer(s) with a newly-shaped conv weight, "
                  f"randomly initialized: {', '.join(skipped_layers)}.")
        else:
            model.load_state_dict(torch.load(args.init_weights, map_location=device))
            print(f"Initialized from {args.init_weights}")
    else:
        print("Training from random initialization (--from-scratch).")

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith(("conv1", "conv2", "conv3", "conv4", "conv5", "conv6")):
                param.requires_grad = False
        print("Encoder layers (conv1-conv6) frozen; only the decoder will be fine-tuned.")

    loss_fn = make_loss(args.loss)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
        if val_loader is not None:
            val_loss, val_dice, val_iou = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        else:
            val_loss, val_dice, val_iou = train_loss, train_dice, train_iou

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f} train_dice={train_dice:.4f} train_iou={train_iou:.4f}  |  "
              f"val_loss={val_loss:.6f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}_loss_{val_loss:.6f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best val_loss, saved {ckpt_path}")

    latest_path = os.path.join(args.output_dir, "latest.pt")
    torch.save(model.state_dict(), latest_path)
    print(f"\nDone. Best val_loss: {best_val_loss:.6f}. Latest weights also saved to {latest_path}.")

    if history:
        history_csv_path = os.path.join(args.output_dir, "training_history.csv")
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        if not args.no_plot:
            plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))
    print("Next step: visually compare against production with, e.g.:")
    print(f"    python run_segmenter_inspect.py --weights {args.output_dir}/<best_checkpoint>.pt "
          f"--input-dir problematic_raw_images/")
    if args.skip_connections:
        print("(run_segmenter_inspect.py auto-detects this is a SegmenterModelSkip checkpoint "
              "from its tensor shapes - no extra flag needed.)")
    print("Only copy the new checkpoint over EmbryoSegmenter/weights/ once you're satisfied "
          "it's at least as good on your held-out/problematic images.")


if __name__ == "__main__":
    main()
