# FEMI Codebase Guide

This document is a walkthrough of the FEMI repository for engineers who are
new to the project. It complements the inline code comments that have been
added throughout the codebase (see each `.py` file for detailed,
line-by-line explanations) and the [README](README.md) (which covers
installation and CLI usage).

For a deep dive specifically on the embryo segmentation preprocessing
step, see:
- [`SEGMENTER_PERFORMANCE_ANALYSIS.md`](SEGMENTER_PERFORMANCE_ANALYSIS.md) — asymptotic complexity / efficiency analysis.
- [`SEGMENTER_ACCURACY_ROADMAP.md`](SEGMENTER_ACCURACY_ROADMAP.md) — root-cause diagnosis and options for improving segmentation accuracy (YOLO vs. more data vs. other fixes).

## 1. What is FEMI?

FEMI is a **foundation model for embryology**: a Vision Transformer Masked
Autoencoder (ViT-MAE, via HuggingFace `transformers`) that is pretrained in
a self-supervised way on large amounts of unlabeled IVF (in-vitro
fertilization) time-lapse embryo images. Once pretrained, FEMI's learned
visual representations can be **fine-tuned** on smaller labeled datasets
for specific clinical prediction tasks, such as:

- Binary classification (e.g. predicting embryo viability/implantation).
- Regression (e.g. predicting a continuous quality score).

It supports both single-image inputs and short video clips (sequences of
time-lapse frames) as input, and can optionally incorporate maternal age as
an additional input feature.

The repository is written for **TensorFlow/Keras** for the main model
(pretraining and fine-tuning), but the standalone embryo image
preprocessing/segmentation step is written in **PyTorch**.

## 2. Repository Structure

```
FEMI/
├── README.md                      # Installation + CLI usage instructions
├── requirement.txt                # Python dependency pins
├── main_pretrain.py               # Self-supervised pretraining entry point (MAE)
├── main_finetune.py               # Supervised fine-tuning entry point (classification/regression)
├── utils/
│   └── helpers.py                 # Shared TF/Keras utilities: LR schedules, callbacks, custom layers
└── preprocessing/
    ├── preprocessing_images.py    # Standalone script: crop raw images around the embryo
    └── EmbryoSegmenter/
        └── SegmenterModel.py      # PyTorch CNN model used to segment the embryo
```

There is no `weights/` directory checked into the repo; pretrained FEMI
weights and the segmenter's weights file
(`EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt`) must be
downloaded/placed separately (see README's HuggingFace link for FEMI
weights).

## 3. File-by-File Breakdown

### `README.md`
The user-facing documentation: how to install dependencies, how to
pretrain, how to fine-tune (including the exact expected data/CSV
formats), and a brief pointer to the preprocessing script. Read this first
if you just want to *run* the existing pipelines.

### `requirement.txt`
Pinned Python package versions (numpy, pandas, pillow, opencv-python,
tensorflow, torch, transformers, openpyxl). Note that both TensorFlow (for
the main model) and PyTorch (for the segmenter) are required if you intend
to run the full pipeline, including preprocessing.

### `main_pretrain.py`
**Purpose:** Continue self-supervised pretraining of FEMI on your own
unlabeled image dataset using the Masked Autoencoder (MAE) objective —
random patches of each image are masked out and the model is trained to
reconstruct them. No labels are required, just a flat folder of images.

**Key pieces:**
- `get_args_parser()` — defines all CLI flags (batch size, epochs, learning
  rate/warmup schedule, data augmentation toggle, device/GPU config, etc).
- `main(args)`:
  1. Configures GPU visibility and decides whether to use
     `tf.distribute.MirroredStrategy` for multi-GPU training.
  2. Builds a `tf.data.Dataset` pipeline (`process_path` /
     `process_path_aug`) that reads image files, resizes to 224x224,
     normalizes with ImageNet mean/std, optionally augments (random crop +
     horizontal flip), and batches/prefetches.
  3. Builds a `WarmUpCosine` learning-rate schedule (from
     `utils/helpers.py`).
  4. Loads the base FEMI checkpoint (`TFViTMAEForPreTraining.from_pretrained`)
     and compiles it with `AdamW`.
  5. Trains with `EarlyStopping` and (optionally) per-epoch checkpointing.
  6. Saves final weights to `--output_dir`.

**Run with:** `python main_pretrain.py --data_path data --femi_model_path
tf_femi ...` (see README for full example).

### `main_finetune.py`
**Purpose:** Supervised fine-tuning of a pretrained FEMI checkpoint on a
labeled dataset for either binary classification or regression, using
either single images or short videos per embryo, with an optional maternal
age input.

**Key pieces:**
- `get_args_parser()` — CLI flags for task type
  (`--do_binary_classification` / `--do_regression`), input type
  (`--video_input`, `--num_frames_in_vid`), age usage (`--use_age`),
  encoder-unfreezing strategy (`--num_transformer_blocks_unfreeze`),
  learning-rate schedule/plateau options, and standard dataset/device args.
- `main(args)`:
  1. Device + reproducibility setup (fixes all RNG seeds).
  2. Resolves the task/input configuration flags into simple booleans used
     throughout the rest of the function.
  3. **`IVFDataset(Sequence)`** — a Keras data generator that lazily reads
     one batch of embryo image(s)/video(s) + labels (+ age) from disk at a
     time, so the whole dataset never needs to fit in memory at once.
  4. **`create_model_linear_probe()`** — builds the actual fine-tuning
     model:
     - Re-implements FEMI's ViT encoder forward pass manually (patch
       embed → transformer encoder → layernorm → drop CLS token → global
       average pool) as a reusable `feature_extractor` sub-model.
     - Freezes the whole encoder ("linear probe") or selectively unfreezes
       the last N transformer blocks, depending on
       `--num_transformer_blocks_unfreeze`.
     - For video input: wraps the feature extractor in `TimeDistributed`,
       then summarizes the per-frame feature sequence with a Bidirectional
       LSTM + a custom attention-pooling mechanism
       (`AttentionWithContext` + `Addition`, from `utils/helpers.py`).
     - Optionally fuses a maternal-age embedding into the final features.
     - Adds a small classification (sigmoid) or regression (linear) head
       and compiles the model with the appropriate loss/metrics.
  5. Loads and validates the labels CSV, splits into train/validation.
  6. Builds (optionally multi-GPU) and trains the model with early
     stopping, LR warmup, and optional `ReduceLROnPlateau`.
  7. Saves final fine-tuned weights to `--output_dir` (prefixed `ft-`).

**Run with:** `python main_finetune.py --data_path data --csv_path
data.csv --femi_model_path tf_femi --do_binary_classification 1 ...` (see
README for more examples, including video + age).

### `utils/helpers.py`
Shared TensorFlow/Keras building blocks used by both training scripts:

- `WarmUpCosine` — a step-based learning rate schedule: linear warmup
  followed by cosine decay to 0. Used in `main_pretrain.py`.
- `WarmupScheduler` — an epoch-based linear-warmup Keras `Callback`. Used
  in `main_finetune.py`.
- `CustomReduceLROnPlateau` — like Keras' built-in `ReduceLROnPlateau`, but
  additionally restores the best-seen weights every time it reduces the
  learning rate (not just at the end of training).
- `get_lr_metric(optimizer)` — exposes the optimizer's live learning rate
  as a trackable Keras metric (for logging/monitoring only).
- `AttentionWithContext` / `Addition` — a learned attention-pooling
  mechanism (Bahdanau-style additive attention with a trainable context
  vector) used to combine a sequence of per-frame video features into a
  single embryo-level feature vector in `main_finetune.py`'s video path.
- `set_trainable_recursively(layer, trainable, ...)` — recursively freezes
  or unfreezes a Keras layer and all its sublayers, used to selectively
  unfreeze specific ViT transformer blocks during fine-tuning.

### `preprocessing/preprocessing_images.py`
**Purpose:** A standalone script (not a function library — it runs
top-to-bottom on import) that preprocesses a directory of raw embryo
images before they're used for FEMI pretraining/fine-tuning. It:

1. Loads + resizes each image to 224x224 (`load_image`).
2. Runs it through the PyTorch `SegmenterModel` to get a binary mask of
   "which pixels belong to the embryo" (`segment_image`).
3. Filters the mask's contours to keep only sufficiently circular ones
   (`calculate_circularity`), since embryos are roughly circular and this
   removes noise/artifacts.
4. If that produces an empty mask, retries once after automatically
   correcting brightness/contrast (`automatic_brightness_and_contrast`).
5. If it's *still* empty, falls back to keeping the whole (uncropped)
   image rather than discarding it.
6. Crops to the final mask's bounding box (`find_bounding_box`), resizes
   back to 224x224, and saves the result.

Images are processed in parallel via
`concurrent.futures.ProcessPoolExecutor` (`process_image` is the unit of
work distributed across processes).

**Before running:** you must edit the placeholder values of `images_dir`
and `cropped_images_dir` near the bottom of the file to point at your
actual input/output directories.

**Note:** This script's top-level code (model loading, `os.listdir`, and
the `ProcessPoolExecutor` call) executes immediately on import without an
`if __name__ == "__main__":` guard. This works correctly on Linux (which
uses the `fork` multiprocessing start method by default) but is fragile on
platforms using the `spawn` start method — see
[`SEGMENTER_PERFORMANCE_ANALYSIS.md`](SEGMENTER_PERFORMANCE_ANALYSIS.md)
for details and a suggested fix.

### `preprocessing/EmbryoSegmenter/SegmenterModel.py`
**Purpose:** Defines the small PyTorch CNN (`SegmenterModel`) used by
`preprocessing_images.py` to predict the embryo segmentation mask.

**Architecture:** A 6-layer convolutional encoder that downsamples a
224x224x1 grayscale frame down to a 7x7x512 bottleneck (doubling channels
and halving spatial resolution at each of 5 max-pool steps), followed by a
mirrored 6-layer decoder that upsamples back to 224x224x1 using simple
(non-learned) interpolation, ending in a sigmoid activation that outputs a
per-pixel "embryo probability" mask. There are **no skip connections**
between encoder and decoder (unlike a classic U-Net), which keeps the
model small and fast but may limit fine boundary precision — see the
performance analysis doc for more on this trade-off.

Pretrained weights for this model are expected at
`EmbryoSegmenter/weights/epoch_102_loss_0.098710.pt` (not included in the
repo).

### `preprocessing/run_segmenter_inspect.py`
**Purpose:** A standalone CLI tool (not part of the original repo's
production pipeline) for interactively running *just* the segmenter — no
TensorFlow/transformers needed — on a handful of images and inspecting
every intermediate artifact (resized input, raw mask, filtered mask,
bounding-box overlay, final crop) plus diagnostic flags (fallback
triggers, crop-area fraction, border-touching). Mirrors
`process_image()`'s logic exactly so what you see matches production
behavior. See the README's "Running just the segmenter on a few images"
section for setup/usage, and `SEGMENTER_ACCURACY_ROADMAP.md` for how to
interpret the diagnostics.

## 4. How the Pieces Fit Together (End-to-End Workflow)

```
Raw images
   │
   ▼
preprocessing_images.py  ──uses──▶  SegmenterModel.py (PyTorch)
   │  (crops each image tightly around the embryo)
   ▼
Cropped 224x224 images
   │
   ├─────────────────────────────┐
   ▼                              ▼
main_pretrain.py            main_finetune.py
(self-supervised MAE          (supervised classification/
 pretraining on unlabeled       regression on labeled data,
 images)                        starting from a FEMI checkpoint)
   │                              │
   ▼                              ▼
Updated FEMI checkpoint      Task-specific fine-tuned model
(can be fed back into        (final deliverable for a given
 main_finetune.py)            clinical prediction task)
```

In short: **segment/crop → pretrain (optional, self-supervised) → fine-tune
(supervised, task-specific)**.

## 5. Suggested Reading Order for a New Engineer

1. `README.md` — get the big picture and CLI usage.
2. This document (`CODEBASE_GUIDE.md`).
3. `preprocessing/EmbryoSegmenter/SegmenterModel.py` — smallest, self-contained file.
4. `preprocessing/preprocessing_images.py` — see how the segmenter is used in practice.
5. `utils/helpers.py` — shared building blocks referenced by both training scripts.
6. `main_pretrain.py` — simpler of the two training scripts (no custom model architecture).
7. `main_finetune.py` — most complex file; ties everything together (custom model head, video/attention path, age fusion).
8. `SEGMENTER_PERFORMANCE_ANALYSIS.md` and `SEGMENTER_ACCURACY_ROADMAP.md` — for anyone working on preprocessing performance/accuracy.
