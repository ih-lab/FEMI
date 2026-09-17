"""
train_segmenter.py

Training/fine-tuning script for `EmbryoSegmenter/SegmenterModel.py`. This
did not exist before — there was no training pipeline for this component
in the repo (see SEGMENTER_ACCURACY_ROADMAP.md, "no existing segmenter
training script or dataset"). This fills that gap, and is designed to
plug directly into `labelme_json_to_mask.py`'s output convention:
  - masks are single-channel 0/255 PNGs named "<image_stem>_mask.png"
    (configurable suffix), at the ORIGINAL image resolution (not resized).
  - this script resizes each image/mask pair together at load time
    (image: bilinear, mask: nearest, to keep the mask binary), so mask and
    image always end up pixel-aligned regardless of source resolution.

Handling "some existing masks are inaccurate": pass MULTIPLE
--images-dir/--masks-dir pairs (or --manifest rows). Pairs are matched by
file stem, and later sources OVERRIDE earlier ones for the same stem. So
to replace old, inaccurate masks with newly corrected ones, just list your
corrected masks directory LAST:

    python train_segmenter.py \\
        --images-dir /path/to/old/raw_images --masks-dir /path/to/old/masks \\
        --images-dir /path/to/new/raw_images --masks-dir preprocessing/masks

Only run this on machines/environments where you're comfortable running
model training (needs more compute than the inference-only tools in this
directory, though the model is still small — see SEGMENTER_PERFORMANCE_ANALYSIS.md).
No new dependencies beyond what's already in requirements-segmenter.txt
(torch, numpy, pillow) — OpenCV is not needed for training.

Fidelity to the original method (Rajendran et al. 2025, "A foundational
model for in vitro fertilization trained on 18 million time-lapse
images", Nature Communications, Methods -> "Datasets and
preprocessing"): the paper states the original segmenter was an
"InceptionV3 U-Net" trained on 30 image-mask pairs with an 80/20 train/val
split and "random cropping, flipping, rotation, and translation"
augmentation. This script matches the split ratio (--val-frac default
0.2) and augmentation types (crop/flip/rotation; "translation" is
approximated via the crop's random offset - see SegmentationDataset). It
does NOT reproduce two things: (1) the paper's "InceptionV3 U-Net" implies
skip connections, but `SegmenterModel.py` in this repo has none (a
pre-existing architecture gap unrelated to this training script - see
SEGMENTER_ACCURACY_ROADMAP.md); (2) the paper does not state a loss
function, optimizer, learning rate, or epoch count for the segmenter
specifically, so --loss/--lr/--epochs below are reasonable defaults, not
values taken from the paper. Iteratively fine-tuning an existing
checkpoint on newly corrected masks (this script's core use case) is not
described in the paper at all - it only documents one training run on the
original 30 pairs.

Using the ORIGINAL 30-pair dataset (if you have it): unlike
labelme_json_to_mask.py's output, the original dataset's masks are
typically plain JPEGs named identically to the raw image (no "_mask"
suffix, e.g. "foo.jpg" raw + "foo.jpg" mask in a separate folder), not
"_mask.png". Use --mask-suffix "" --mask-ext .jpg to match that layout.
Mask pixel values don't need to be clean 0/255 - SegmentationDataset
resizes masks with nearest-neighbor (introducing no new intermediate
values) and thresholds at 127 when building the training tensor, so mild
JPEG compression artifacts in the source mask do not affect training. If
the original data also came with its own pre-defined train/validation
split (e.g. separate "mask" and "mask_valid" folders, matching the
paper's 80/20 split), pass the validation folder(s) via
--val-images-dir/--val-masks-dir to use that split directly instead of
--val-frac's random shuffle - see example 4 below.

Usage:
    # Sanity-check your data pairing/alignment before spending time training:
    python train_segmenter.py --images-dir raw/ --masks-dir preprocessing/masks \\
        --preview-only --preview-dir preview/

    # Fine-tune from the current production weights (recommended default):
    python train_segmenter.py --images-dir raw/ --masks-dir preprocessing/masks \\
        --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned

    # Train from scratch instead:
    python train_segmenter.py --images-dir raw/ --masks-dir preprocessing/masks \\
        --from-scratch --epochs 200 --output-dir EmbryoSegmenter/weights_finetuned

    # Original 30-pair dataset + new Labelme labels (different mask naming):
    python train_segmenter.py \\
        --images-dir mask/psd --masks-dir mask/Embryo \\
        --images-dir new_labels/raw_images --masks-dir new_labels/masks \\
        --mask-suffix "" --mask-suffix _mask \\
        --mask-ext .jpg --mask-ext .png \\
        --val-images-dir mask_valid/psd --val-masks-dir mask_valid/Embryo \\
        --freeze-encoder \\
        --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned

After training, use run_segmenter_inspect.py with `--weights
path/to/new/checkpoint.pt` to visually compare the new model against the
current production one on a handful of images BEFORE promoting it (i.e.
before copying it over EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt).
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

from EmbryoSegmenter.SegmenterModel import SegmenterModel

DEFAULT_INIT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "EmbryoSegmenter", "weights", "epoch_102_loss_0.098710.pt",
)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Pair discovery: match raw images to masks by filename stem, across one or
# more (images_dir, masks_dir) sources, plus an optional manifest CSV.
# Later sources win on stem collisions - this is the mechanism for
# "override inaccurate old masks with corrected new ones".
# ---------------------------------------------------------------------------

def _expand_per_pair(values, n_pairs, flag_name, default):
    """Expand a CLI list to one entry per (images_dir, masks_dir) pair.

    - None or a single value -> broadcast to all pairs.
    - len == n_pairs -> use as-is (one convention per pair).
    - otherwise -> raise with a helpful message.
    """
    if values is None:
        values = [default]
    if len(values) == 1:
        return values * n_pairs
    if len(values) != n_pairs:
        raise ValueError(
            f"--{flag_name} given {len(values)} time(s) but there are {n_pairs} "
            f"--images-dir/--masks-dir pair(s). Pass it once (applies to all pairs) "
            f"or once per pair, in the same order."
        )
    return values


def discover_pairs(images_dirs, masks_dirs, mask_suffixes=None, mask_exts=None):
    """Match raw images to masks by filename stem within each (images_dir,
    masks_dir) pair, expecting masks named "<stem><mask_suffix><mask_ext>".

    mask_suffixes/mask_exts may be a single value (broadcast to every pair)
    or a per-pair list aligned with images_dirs. Defaults (_mask, .png)
    match labelme_json_to_mask.py. The original paper dataset uses ("", .jpg).
    """
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
    Loads (raw image, binary mask) pairs, resizing both to 224x224 at
    __getitem__ time (image: bilinear: mask: nearest, to preserve a clean
    0/1 mask regardless of source resolution - matches the convention
    documented in labelme_json_to_mask.py).

    Augmentation (train split only) is applied to the ORIGINAL-resolution
    image+mask pair, before resizing, and is chosen to match the four
    augmentation types described for the original segmenter training in
    the FEMI paper (Rajendran et al. 2025, Methods -> "Datasets and
    preprocessing"): "random cropping, flipping, rotation, and
    translation":
      - random crop (a random sub-region, `crop_min_scale`-1.0 of the
        original size, at a random offset) - this also stands in for
        "translation", since cropping from a random offset shifts the
        embryo's position within the eventual 224x224 frame, similar in
        effect to translating the frame. The paper does not specify exact
        crop/translation magnitudes, so this is an approximation, not a
        byte-for-byte reproduction.
      - horizontal/vertical flip
      - 90-degree-multiple rotation only (no arbitrary-angle rotation,
        which would need interpolation and blur the mask's binary edges)
    Brightness/contrast jitter is applied to the image only, after
    resizing, since it should never affect the mask - the paper does not
    list this as a training-time augmentation (it describes brightness/
    contrast correction as a separate, unconditional PREPROCESSING step
    applied to every image, not a random training-time augmentation), so
    treat this as an extra addition beyond what's described in the paper.
    """

    def __init__(self, pairs, size=(224, 224), augment=False, crop_min_scale=0.85):
        self.pairs = pairs  # list of (image_path, mask_path)
        self.size = size
        self.augment = augment
        self.crop_min_scale = crop_min_scale

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
        img, mask = self._random_crop(img, mask)  # approximates paper's "cropping" + "translation"
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
            # Should not normally happen if the mask was produced by
            # labelme_json_to_mask.py against this exact image, but resize
            # defensively rather than crash on a mismatched pair.
            mask = mask.resize(img.size, Image.NEAREST)

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
    """Soft Dice loss on already-sigmoid-activated predictions. Handles the
    embryo/background class imbalance (embryo is often well under half the
    frame) better than BCE alone, which can be dominated by the easy
    "predict background" majority class."""

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
        description="Train/fine-tune SegmenterModel on (raw image, binary mask) pairs, "
                    "e.g. masks produced by labelme_json_to_mask.py.")
    parser.add_argument("--images-dir", action="append", required=True,
                        help="Directory of raw images. Repeatable; pair with --masks-dir "
                             "in the same order. Later --images-dir/--masks-dir pairs "
                             "OVERRIDE earlier ones on filename-stem collisions (use this "
                             "to replace old inaccurate masks with corrected ones).")
    parser.add_argument("--masks-dir", action="append", required=True,
                        help="Directory of mask PNGs matching --images-dir, named "
                             "'<image_stem><mask-suffix>.png' (see --mask-suffix).")
    parser.add_argument("--mask-suffix", action="append", default=None,
                        help="Mask filename suffix (repeatable, one per --images-dir pair). "
                             "Default if omitted: _mask (labelme_json_to_mask.py convention, "
                             "e.g. foo.jpg -> foo_mask.png). Pass '' for the original paper "
                             "dataset (mask named identically to the image). When combining "
                             "legacy + Labelme data, pass once per pair, e.g. "
                             "--mask-suffix '' --mask-suffix _mask.")
    parser.add_argument("--mask-ext", action="append", default=None,
                        help="Mask file extension (repeatable, one per --images-dir pair). "
                             "Default if omitted: .png. Pass '.jpg' for the original paper "
                             "dataset's lossy JPEG masks. When combining legacy + Labelme "
                             "data, pass once per pair, e.g. --mask-ext .jpg --mask-ext .png.")
    parser.add_argument("--val-mask-suffix", action="append", default=None,
                        help="Validation mask suffix (repeatable, one per --val-images-dir "
                             "pair). Default when --val-images-dir is given: '' (legacy "
                             "mask_valid/ layout). Ignored if --val-images-dir is not used.")
    parser.add_argument("--val-mask-ext", action="append", default=None,
                        help="Validation mask extension (repeatable). Default when "
                             "--val-images-dir is given: .jpg. Ignored if --val-images-dir "
                             "is not used.")
    parser.add_argument("--manifest", default=None,
                        help="Optional CSV with columns image_path,mask_path for pairs "
                             "that don't fit the directory+suffix convention. Applied "
                             "after --images-dir/--masks-dir (so it can also override).")
    parser.add_argument("--val-images-dir", action="append", default=[],
                        help="Directory of raw images reserved for validation ONLY - "
                             "repeatable, pairs with --val-masks-dir in order, same "
                             "--mask-suffix/--mask-ext convention as --images-dir. If given, "
                             "this EXACT set is used as the validation split (all "
                             "--images-dir/--masks-dir pairs are used for training) instead "
                             "of --val-frac's random shuffle - use this to preserve a "
                             "pre-existing train/val split (e.g. the original paper "
                             "dataset's separate 'mask' and 'mask_valid' folders).")
    parser.add_argument("--val-masks-dir", action="append", default=[],
                        help="Mask directory matching --val-images-dir, in order.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of --images-dir/--masks-dir pairs held out for "
                             "validation via random shuffle (default: 0.2, matching the "
                             "80/20 split reported for the original segmenter training in "
                             "Rajendran et al. 2025). Ignored if --val-images-dir is given.")
    parser.add_argument("--crop-min-scale", type=float, default=0.85,
                        help="Random-crop augmentation lower bound, as a fraction of the "
                             "original image/mask size (default: 0.85, i.e. crop between 85%% "
                             "and 100%% of the frame at a random offset). Approximates the "
                             "paper's 'random cropping'+'translation' augmentations; set to 1.0 "
                             "to disable cropping while keeping flip/rotation augmentation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce_dice",
                        help="Loss function (default: bce_dice - a 50/50 mix, generally more "
                             "robust to embryo/background class imbalance than BCE alone).")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help="Checkpoint to initialize from for fine-tuning "
                             f"(default: current production weights at {DEFAULT_INIT_WEIGHTS}).")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore --init-weights and start from random initialization.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers (conv1-conv6), fine-tune only the decoder. "
                             "Recommended if you only have a handful of new corrected examples, "
                             "to reduce the risk of overfitting/forgetting on the encoder.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable train-time augmentation (flips/rotation/brightness jitter).")
    parser.add_argument("--output-dir", default="EmbryoSegmenter/weights_finetuned",
                        help="Where to save checkpoints (default: EmbryoSegmenter/weights_finetuned - "
                             "deliberately NOT the production weights/ dir; promote manually after "
                             "validating with run_segmenter_inspect.py).")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-only", action="store_true",
                        help="Just discover pairs, write preview overlays, and exit "
                             "(no training) - use this FIRST to sanity-check alignment.")
    parser.add_argument("--preview-dir", default="preview",
                        help="Directory for preview overlays (default: ./preview).")
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
        preview_dataset = SegmentationDataset(train_pairs + val_pairs, augment=False)
        save_preview(preview_dataset, args.preview_dir, n=args.preview_n)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SegmentationDataset(train_pairs, augment=not args.no_augment,
                                    crop_min_scale=args.crop_min_scale)
    val_ds = SegmentationDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if val_pairs else None

    model = SegmenterModel().to(device)
    if not args.from_scratch:
        if not os.path.exists(args.init_weights):
            print(f"ERROR: --init-weights not found at {args.init_weights}. "
                  f"Pass --from-scratch to train from random initialization instead.",
                  file=sys.stderr)
            sys.exit(1)
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

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
        if val_loader is not None:
            val_loss, val_dice, val_iou = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        else:
            val_loss, val_dice, val_iou = train_loss, train_dice, train_iou

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f} train_dice={train_dice:.4f} train_iou={train_iou:.4f}  |  "
              f"val_loss={val_loss:.6f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}_loss_{val_loss:.6f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best val_loss, saved {ckpt_path}")

    latest_path = os.path.join(args.output_dir, "latest.pt")
    torch.save(model.state_dict(), latest_path)
    print(f"\nDone. Best val_loss: {best_val_loss:.6f}. Latest weights also saved to {latest_path}.")
    print("Next step: visually compare against production with, e.g.:")
    print(f"    python run_segmenter_inspect.py --weights {args.output_dir}/<best_checkpoint>.pt "
          f"--input-dir problematic_raw_images/")
    print("Only copy the new checkpoint over EmbryoSegmenter/weights/ once you're satisfied "
          "it's at least as good on your held-out/problematic images.")


if __name__ == "__main__":
    main()
