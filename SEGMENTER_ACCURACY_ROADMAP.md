# Embryo Segmenter: Accuracy Improvement Roadmap

**Audience:** whoever picks up the "improve the embryo segmenter" task next.
**Companion doc:** see
[`SEGMENTER_PERFORMANCE_ANALYSIS.md`](SEGMENTER_PERFORMANCE_ANALYSIS.md)
for the efficiency/complexity analysis this roadmap builds on.

## Status (as of September 2026)

Work so far lives in **`preprocessing/run_segmenter_inspect.py`** — a
standalone inspection CLI for manually testing the segmenter on a handful
of images. **`preprocessing/preprocessing_images.py` (production batch
preprocessing) still uses the original circularity-union logic** until the
Phase 1 fixes below are ported over.

| Item | Status | Where |
|---|---|---|
| Phase 0 diagnostics (crop fraction, border touch, fallback flags, mask/bbox overlays) | **Done** | `run_segmenter_inspect.py` |
| Manual-review step: copy flagged images' mask/overlay/crop into `needs_review/` + `needs_review.csv` | **Done (inspect tool only)** | `run_segmenter_inspect.py --review-on` (default: full-image fallback only) |
| Phase 1(a): square-up crop before resize (`squarify_bbox`) | **Done (inspect tool only)** | `_6_cropped_224_fixed.png` output |
| Phase 1(b): single best-contour selection (no circularity, no union) | **Done (inspect tool only)** | `select_best_contour()` |
| Phase 1(b): reject border-touching / oversized contours | **Done (inspect tool only)** | `select_best_contour()` defaults: `border_margin=2`, `max_area_frac=0.9` |
| Phase 1: `possible_background_false_positive` flag (crop area > 70%) | **Done (inspect tool only)** | `--background-area-flag-threshold` (default 0.7) |
| Phase 1(c): morphological closing before contour-finding | **Not yet** | — |
| Phase 1(d): bounding-box safety margin | **Not yet** | — |
| Port Phase 1 fixes into `preprocessing_images.py` | **Not yet** | Next step for production |
| Phase 1(b) noise-speck bug: a tiny non-border-touching speck could beat a much larger, genuine border-touching contour | **Fixed** | `select_best_contour(min_area_frac=0.01)` — see Root cause B |
| Cross-check against the published methodology (Rajendran et al. 2025) | **Done** | §3 below |
| Segmenter training/fine-tuning pipeline | **Done** | `train_segmenter.py` (didn't exist before — see §4 Option B) |
| Skip connections (U-Net variant, root cause C) | **Done, opt-in** | `train_segmenter.py --skip-connections` (`SegmenterModelSkip`); `run_segmenter_inspect.py` auto-detects and loads either architecture. Off by default — see caveat below Root cause C. |
| Labeled dataset: original 30-pair paper dataset recovered + 10 new labeled pairs | **Done** | `mask/` + `mask_valid/` (23 train / 5 val) + `new_labels/` (10 pairs) — 38 total |

### Validated on 7 problematic production images (`problematic_raw_images/`)

Compared against the **old** production-style logic (circularity ≥ 0.7,
union of all passing contours) on the same weights and images:

| Metric | Old logic | New logic (`select_best_contour`) |
|---|---|---|
| Full-image fallback (`np.ones_like`) | 1 / 7 | **0 / 7** |
| Brightness-correction fallback | 1 / 7 | **0 / 7** |
| Possible dish/vignette false-positive (crop > 70%) | 1 / 7 | **0 / 7** |
| Bounding box touches border | 4 / 7 | 4 / 7 *(unchanged — model/source issue)* |
| Likely visible aspect-ratio distortion in `_5` crop | 3 / 7 | 4 / 7 *(unchanged until squarify is used for final output)* |

**Standout fix:** `22987821_0331_110.5_SB_0.jpg` previously hit the
worst-case full-image fallback (crop area 99.1%, essentially no cropping).
The new logic finds the actual embryo on the first pass (crop area 34.6%).

**Still open:** 4 images still have border-touching boxes and/or visible
stretching in the production-faithful `_5` output — these point to root
causes C/D (model under-recall or genuinely clipped source), not the
contour-selection bug. Use `_4_bbox_overlay.png` (red = final box, green =
squarified box) and `_6_cropped_224_fixed.png` to review.

---

**Reported problems** (four distinct symptom categories observed in
production):
1. **Hatching / irregularly-shaped embryos** get cropped incorrectly (part
   of the embryo missed).
2. **The petri dish background or the microscope's circular light/field-of-
   view is mistaken for the embryo**, resulting in little-to-no cropping
   (the image is preserved largely as-is).
3. **The embryo is not preserved as a whole**, usually missing the bottom
   portion — sometimes because the *source* image itself never fully
   captured the embryo (an acquisition-time issue, not a preprocessing
   bug).
4. **The final cropped output looks visibly stretched/warped** — the
   embryo's round shape is distorted because a non-square bounding box is
   force-resized to 224×224 without preserving aspect ratio (root cause F).

This document lays out (1) a key architectural fact that reframes the
problem, (2) a diagnosis mapping each of these three symptoms to specific,
identifiable root causes in the current code, (3) an evaluation of
"switch to YOLO" vs. "get more/better labeled data" vs. other options, and
(4) a recommended phased roadmap.

## 1. Important reframing: this is a *detection* problem, not a *segmentation* problem

Look closely at `process_image()` in `preprocessing_images.py`:

```224:225:preprocessing/preprocessing_images.py
    if np.max(img) > 0:
        # Segment the image
        segmented_images, mask = segment_image(img)
```

`segment_image()` returns two things: `segmented_images` (the original
pixels multiplied by the mask) and `mask` (the raw binary mask). **Only
`mask` is ever used again** — to find contours, filter by circularity, and
compute a bounding box. `segmented_images` is discarded. The final saved
output (`cropped_image`) is a crop of the **original, unmasked** image
using that bounding box:

```286:290:preprocessing/preprocessing_images.py
        min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
        cropped_image = img[min_y:max_y, min_x:max_x]    
        cropped_image = Image.fromarray(cropped_image)
        cropped_image = cropped_image.resize((224, 224))
        cropped_image.save(cropped_images_dir+image)
```

**Implication:** despite using a full pixel-wise segmentation network, the
pipeline only ever needs a rectangular bounding box. No downstream
consumer needs a pixel-accurate mask. This matters a lot for choosing a
fix, because:

- A bounding-box regression task (4 numbers) is strictly lower-dimensional
  and generally more sample-efficient/robust to train than a dense
  per-pixel mask (50,176 outputs), especially with limited labeled data.
- It opens the door to using an **object detector** (YOLO or otherwise)
  instead of a segmentation network, and/or to simplifying post-processing.
- It also means **existing pixel-mask training data can be converted to
  bounding-box training data for free** (just take the bounding box of the
  existing ground-truth mask) — no new manual annotation is required to
  bootstrap a detector-based approach.

## 2. Root-cause diagnosis, mapped to each reported symptom

There are several independent places this can go wrong, and they call for
different fixes. **Before picking a solution, it's worth instrumenting
which one is actually happening on your data** (see Phase 0 below) —
otherwise you risk spending weeks retraining a model when the real bug is
a few lines of post-processing code.

### Root cause A — the circularity filter takes the UNION of every passing contour, not the single best one

**Note:** this is not merely a hypothesized bug — §3 confirms the
published methodology itself calls for picking **the single most
circular contour**, so this "union of all passing contours" behavior is
a deviation from the paper's own described design, not a defensible
alternative design choice.

This is the code in question:

```257:262:preprocessing/preprocessing_images.py
        min_circularity = 0.7  # This is a threshold for circularity
        circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
        # Create an empty mask for the final output
        final_mask = np.zeros_like(cv2mask)
        cv2.drawContours(final_mask, circular_contours, -1, (255), thickness=cv2.FILLED)
```

`cv2.drawContours(..., -1, ...)` draws **every** contour in
`circular_contours` onto `final_mask`, and the final bounding box is then
computed over that combined region. Two separate bugs fall out of this:

- **Symptom 2 (dish/microscope light treated as the embryo, little
  cropping):** a petri dish rim or the microscope's circular field-of-view
  vignette is very frequently an almost-perfect circle spanning most of
  the frame — exactly the shape `calculate_circularity` scores highest
  (a perfect circle scores ~1.0, comfortably clearing the 0.7 threshold).
  If the model's mask includes any trace of that rim/vignette as a
  separate contour (common in poor-focus or low-signal frames where the
  embryo itself doesn't stand out much), it will pass the filter and get
  **unioned into `final_mask` right alongside the real embryo contour (or
  in place of it)**. Since the rim/vignette contour is close to the size
  of the whole frame, the resulting bounding box balloons to cover almost
  the entire image — matching the "barely cropped" symptom exactly. This
  is a **specific, identifiable, code-level bug**, not a fuzzy model
  quality issue: the algorithm has no logic to reject a contour just
  because it's implausibly large or touches the image border.
- **Symptom 1 (hatching/irregular embryos missed):** the flip side of the
  same design — if the true embryo's contour is irregular enough to score
  *below* 0.7 (very plausible for a hatching embryo breaking out of its
  round zona pellucida, or a multi-cell/blastomere-stage embryo), it's
  excluded from the union entirely, and only some other passing contour
  (if any) defines the box.

### Root cause B — no size/position sanity checks on candidate contours

Related to A: there's currently no check that a candidate contour's area
falls within a plausible range for "an embryo in a 224×224 frame," nor any
check on whether it touches/is near the image border, nor any preference
for the contour closest to the image center. A dish rim or light vignette
almost always (a) is far larger than a typical embryo relative to the
frame and (b) touches or nearly touches the image border — both are easy,
cheap signals to reject it that the current code doesn't use at all.

**Noise-speck corollary (found & fixed while comparing `SegmenterModelSkip`
against production on `92442871_0393_134.8_HB_0.jpg`):** `select_best_contour()`'s
border-preference rule ("prefer non-border-touching candidates over
border-touching ones") only falls back to considering border-touching
candidates when the preferred pool is completely *empty*. That meant a
single tiny (e.g. 4-pixel) spurious blob elsewhere in the frame — which
happens not to touch the border — was enough to make it the *entire*
preferred pool, unconditionally beating a >5000x larger, genuine embryo
contour that was only disqualified for grazing the frame edge by 1-2
pixels. Confirmed by direct contour inspection on the raw mask (1 real
contour + 4 near-zero-area specks). **Fixed** by discarding contours below
`min_area_frac` (default 1% of frame area) *before* applying the
border/size preference logic, so a noise speck can no longer win by
default — the real, larger contour still correctly falls back to being
selected when it's the only substantial candidate, border-touching or not.

### Root cause C — the model's raw mask recall is genuinely incomplete

Even before contour filtering, the network may under-predict the embryo's
true extent (contributes most directly to **Symptom 3**, bottom-of-embryo
missing). Plausible contributing factors given the architecture (see
`SegmenterModel.py`):
- **No skip connections** between encoder and decoder, despite the paper
  calling this component a "U-Net" (see §3, item #1 — and note it's also
  not a real Inception-module network) — all boundary detail must be
  reconstructed from a 7×7×512 bottleneck, which is known to hurt fine
  boundary precision and can cause systematic under-prediction at edges
  (plausibly worse at the bottom/edges of frame if there's any positional
  bias in lighting or training data). Adding skip connections to the
  *existing* lightweight architecture is measured at **+28% CPU inference
  time (+9 ms/image), +12.5% parameters** (see
  [`SEGMENTER_PERFORMANCE_ANALYSIS.md`](SEGMENTER_PERFORMANCE_ANALYSIS.md)
  §3.1) — cheap, and most of the current production checkpoint's weights
  can be reused (only 5 conv-weight tensors change shape; 67 of ~72 weight
  tensors transfer directly, including the encoder and all BatchNorm
  params). **Implemented as an opt-in flag** (`train_segmenter.py
  --skip-connections`, `SegmenterModelSkip`) — see §4 Option C item 3 for
  the caveat that emerged from actually testing it: with only ~38 labeled
  examples and `--freeze-encoder`, the 5 newly-shaped decoder layers start
  from random initialization and had visibly noisier raw output on at
  least one held-out image than the fully-pretrained production decoder
  (see the noise-speck corollary under Root cause B, found via this exact
  comparison) — this is a data/convergence issue, not evidence against
  skip connections as an idea. A full real-Inception backbone (optionally
  ImageNet-pretrained) would cost far more compute (~24M params / ~5.7
  GFLOPs vs. ~3.1M / ~1.2 GFLOPs) and carries even more overfitting risk
  given the same ~38 examples (see §4 Option C item 3 for the full
  comparison) — start with skip connections alone, and only once there's
  enough data/epochs for the new layers to actually converge.
- Fixed 0.5 sigmoid threshold may cut off legitimately-embryo but
  lower-confidence pixels, which are more likely to occur at the
  periphery of the embryo (e.g. a fainter lower boundary due to lighting
  angle).

### Root cause D — the source image itself may already be incomplete (an acquisition problem, not a preprocessing bug)

For **Symptom 3** specifically: if the raw time-lapse frame never fully
captured the embryo (dish/well positioning, focus stack limits, or camera
framing at acquisition time), **no segmentation model — CNN, U-Net, or
YOLO — can recover pixels that were never captured.** This needs to be
diagnosed and handled as a *data quality* problem (flag and exclude, or
fix at the acquisition/imaging-system level), not chased as a model
accuracy problem. Phase 0 below includes a cheap heuristic to help tell
these two cases (model under-segmentation vs. genuinely clipped source)
apart.

### Root cause F — cropping to a non-square box, then force-resizing to a square, distorts the embryo's shape

**Confirmed via `run_segmenter_inspect.py` on real production images** (see
worked examples below). This is independent of everything above — it
happens even when the mask/contour logic picks a perfectly reasonable
region, as long as that region isn't square:

```286:290:preprocessing/preprocessing_images.py
        min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
        cropped_image = img[min_y:max_y, min_x:max_x]    
        cropped_image = Image.fromarray(cropped_image)
        cropped_image = cropped_image.resize((224, 224))
        cropped_image.save(cropped_images_dir+image)
```

`PIL.Image.resize((224, 224))` forces the crop into an exact 224×224
square **without preserving aspect ratio**. Since `find_bounding_box`
returns a tight axis-aligned box around the (often irregular) mask, the
box is very often not square — and whenever it isn't, width and height
get scaled by *different* factors, visibly warping the embryo's true
round shape into an oval/stretched form.

**Real examples from a first batch of 7 production images tested with
`run_segmenter_inspect.py`:**

| Image | Bounding box (w×h) | Aspect ratio | Resize scale (x vs. y) | Result |
|---|---|---|---|---|
| `11673354_0331...` | 207×161 | 1.29 | 1.08× vs. 1.39× | Visibly vertically stretched |
| `24502838_0336...` | 86×147 | 0.585 | 2.61× vs. 1.52× | Visibly horizontally stretched |
| `23099214_0336...` | 164×136 | 1.21 | — | Borderline distortion |
| 4 others in the same batch | ratios 0.96–1.11 | — | — | No visible distortion |

**3 of 7 (43%) real sample images** showed a bounding box asymmetric
enough (>20% off from square) to cause visible distortion. This is a
**very common, high-impact bug** — likely more visually obvious to a
human reviewer than any of root causes A–E, since it directly warps the
embryo's shape (a clinically meaningful feature) rather than just
mis-placing the crop boundary.

**Important nuance:** fixing this (see Phase 1) by padding the box to a
square before resizing removes the *stretching*, but if the original box
is already asymmetric because it's touching the image border (e.g. mask
under-recall cut off one side), squaring it up will necessarily expand
into background/dish area on the opposite side rather than recovering
embryo detail — squarifying the crop and fixing the contour-selection
logic (root causes A/B) are **complementary fixes, not substitutes** for
each other.

### Root cause E — training data coverage/labeling issues

If the original training set for this segmenter under-represents certain
developmental stages (e.g. hatching or multi-cell/blastomere stages, which
are visually much less circular than a 1-cell zygote or blastocyst),
under-represents frames where a dish rim/vignette is visible (i.e. no
"negative" examples teaching the model *not* to include it), off-center
framings, or has inconsistent/imprecise mask annotations, the model will
systematically struggle on those cases regardless of architecture.

### Quick reference: symptom → likely cause(s) → fix

| Symptom | Most likely root cause(s) | Fastest fix |
|---|---|---|
| 1. Hatching/irregular embryo missed | A (irregular contour scores < 0.7, excluded) | **Done in inspect tool:** drop circularity entirely; pick single largest plausible contour (see `select_best_contour()`). Still pending: morphological closing, bbox margin |
| 2. Dish/light circle treated as embryo, minimal cropping | A + B (oversized/border-touching circular contour unioned in) | **Done in inspect tool:** reject border-touching/oversized contours; pick single best candidate; flag crop > 70% as `possible_background_false_positive` |
| 3. Bottom of embryo missing | C (model under-recall) and/or D (source image already incomplete) | Phase 0 to distinguish which; Phase 1 margin as a band-aid; Phase 2 skip connections for C; data-quality audit/exclusion for D |
| 4. Final crop looks visibly stretched/warped (embryo shape distorted) | **F (non-square crop force-resized to a square) — confirmed present in production, affects ~43% of a first sample batch** | **Done in inspect tool:** `squarify_bbox()` + `_6_cropped_224_fixed.png`. **Not yet in production** — `_5` output still mirrors the old non-preserving resize until ported |

## 3. Cross-checking against the published methodology (Rajendran et al. 2025)

This repo is the code release for ["A foundational model for in vitro
fertilization trained on 18 million time-lapse images"](https://www.nature.com/articles/s41467-025-61116-2)
(Rajendran et al., *Nature Communications* 16:6235, 2025). The paper's
Methods section ("Datasets and preprocessing") describes the original
embryo segmenter as follows:

> "An InceptionV3 U-Net was trained using 30 image-mask pairs of
> whole-embryo segmentations. 80% of the dataset was used for training
> the model, with the rest being used for validation. To make the model
> more robust, we added data augmentations to the training data,
> including random cropping, flipping, rotation, and translation... Before
> generating segmentation masks, each image undergoes an automatic
> brightness and contrast adjustment to standardize lighting conditions...
> After generating a segmentation mask for each input image, circular
> contours were detected in the image using OpenCV. A bounding box is
> drawn around **the most circular contour**, which is likely the embryo.
> The image is then cropped around the bounding box and resized to
> 224×224 for input into FEMI."

This section checks the current code — both `preprocessing_images.py`
(production, unmodified by the Phase 1 work above) and
`run_segmenter_inspect.py` (the Phase 1 prototype) — against that
description, and is careful to distinguish gaps that **predate** Phase 1
from ones **introduced by** it.

| # | Paper says | Production `preprocessing_images.py` | `run_segmenter_inspect.py` | Caused by Phase 1 work? |
|---|---|---|---|---|
| 1 | "InceptionV3 **U-Net**" | `SegmenterModel.py` is **neither**: no skip connections (the "U-Net" part), and no real Inception modules either — see note below | Same model file, identical | **No** — pre-existing, inherited from the original repo, unrelated to any fix here |
| 2 | Brightness/contrast correction applied to **every image, unconditionally**, before mask generation | Applied only as a **fallback**, triggered only if the first pass finds nothing (lines 265-269) | Identical fallback-only behavior (lines 329-331) — copied as-is | **No** — pre-existing, and deliberately preserved rather than changed |
| 3 | Bounding box drawn around **the single most circular contour** | Unions **every** contour above a 0.7 circularity threshold (lines 258-262) — already a deviation from the paper's own design before Phase 1 touched anything | `select_best_contour()`: single contour, but scored by border/size/centrality, **not circularity** (per explicit direction to drop it — see root cause A) | **Partially** — Phase 1 fixed the "union" bug (bringing it structurally closer to the paper: one contour, not many) but chose a different selection criterion than "most circular" |
| 4 | "Cropped around the bounding box and resized to 224×224" (aspect-ratio handling not stated either way) | Forces a square resize without preserving aspect ratio (line 289) | `_5` output: byte-identical to production on purpose. `_6` is an *added* alternative | **No conflict confirmed** — the paper doesn't state whether aspect ratio should be preserved, so root cause F is an independently-identified implementation issue, not a documented paper deviation. The `_6` fix is additive only, `_5` stays faithful |
| 5 | "Input images are then normalized using ImageNet mean and standard deviation parameters" | N/A to the segmenter itself | N/A to the segmenter itself | **Not a conflict** — this describes normalizing FEMI's (ViT MAE) input *after* the segmenter's crop, not the segmenter's own input. The segmenter's own input normalization (divide by 255, in both files) is a separate, earlier step the paper doesn't contradict |
| 6 | Segmenter trained on 30 pairs, 80/20 split, augmented with random cropping/flipping/rotation/translation | N/A (training script doesn't exist in this repo/this analysis is inference-only) | N/A | **Informs, doesn't conflict** — `preprocessing/train_segmenter.py` (built separately for fine-tuning with new labeled data) defaults `--val-frac` to 0.2 and its augmentation set (random crop + flip + 90°-rotation) to approximate this description, since no training script previously existed to compare against |

**Item #1 in detail — "InceptionV3" is a misnomer, not just "missing skip
connections":** `SegmenterModel.py`'s `BasicConv2d`/`BasicConv2dSigmoid`
helper blocks are a near-verbatim copy of `torchvision`'s `BasicConv2d`
helper from its real InceptionV3 implementation (the file even says so:
`# A copy of Inception V3 allowing a dynamic number of input channels`,
directly above the model class). That's *all* that was borrowed. What's
missing relative to real InceptionV3: multi-branch **Inception modules**
(parallel 1×1/3×3/5×5/pooling branches concatenated together — this model
has a single conv path per stage instead), factorized convolutions, and
ImageNet-pretrained weights. Real InceptionV3 is ~24M parameters / ~5.7
GFLOPs (299×299 input); this model is ~3.1M parameters / ~1.2 GFLOPs — an
~8×/~5× difference. See §4 Option C item 3 for what it would cost to close
each of these two gaps (skip connections vs. a real Inception backbone)
separately, since they have very different cost/risk profiles.

**Net takeaway:** of the six points checked, only **one (#3)** involves
something the Phase 1 work in this repo actually changed — and even
there, production's pre-Phase-1 behavior (union of all circular contours)
was *already* a deviation from the paper's own stated design before any
of this session's fixes existed. The architecture gap (#1) and the
brightness/contrast timing gap (#2) are untouched, pre-existing gaps
between this repo and the published methodology, inherited as-is by
`run_segmenter_inspect.py` for behavioral fidelity. Root cause F (#4) and
the training-pipeline defaults (#6) aren't paper deviations at all — the
paper is either silent or the new script's defaults were chosen to align
with it.

## 4. Evaluating the proposed options

### Option A: Replace the CNN with YOLO

**Would it plausibly fix the accuracy issue?** Likely **yes, substantially**,
for symptoms 1 and 2 — a detector trained end-to-end to output a single
bounding box (with confidence + non-max suppression) has no circularity
assumption and no "union every candidate region" logic at all, so it
structurally cannot reproduce root causes A/B (it doesn't need to decide
*which* contours to merge — it's trained to directly predict the one
correct box, and a dish rim/vignette is exactly the kind of large,
uninteresting-to-the-detector background region that box-based training
with proper negative/background examples teaches a model to ignore). It's
less certain to fix symptom 3 (root cause C — raw recall, or root cause D
— source data already clipped) unless it's also given better/broader
training data; a detector can't recover embryo area that genuinely isn't
distinguishable in the image, same as any model.

**Would it remain efficient?** Yes, most likely. A nano-scale detector
(e.g. YOLOv8n/YOLO11n) run at a similar ~224–256px input resolution is
roughly the same order of magnitude as the current model:

| | Current `SegmenterModel` | YOLOv8n-class detector |
|---|---|---|
| Parameters | ~3.1M | ~3.2M |
| FLOPs (at 224×224 input, scaled down from published 640×640 figures) | ~1.2 GFLOPs | ~1 GFLOP (order of magnitude) |
| Output | full 224×224 mask | 4 box coordinates (+ confidence) |

Since a detection head is much cheaper than a full decoder that
reconstructs a 224×224 mask, a YOLO-style model doing *only* box
regression could plausibly be **less** compute per image than the current
architecture, not more. Big-O complexity is unaffected either way (still
O(1) per image / O(N) overall, per the companion performance doc) — this
is purely a constant-factor comparison, and it looks favorable.

**Costs / risks:**
- **No YOLO-specific training pipeline exists.** `train_segmenter.py`
  (added since this section was first written — see Option B) trains the
  current mask-based CNN, not a detector; adopting YOLO would still mean
  building a separate training pipeline, likely using the `ultralytics`
  package, a new dependency.
- Bounding-box labels are needed, but as noted in §1, these can be
  **derived for free** from any existing pixel-mask training data used for
  the current model — no new manual annotation required to get started.
- Adds an inference dependency footprint (`ultralytics`) and a
  different model format/loading path than the current raw PyTorch
  `state_dict`.
- If the existing masks used for training were themselves already
  affected by root cause C/D/E (i.e. the ground truth itself
  under-represents part of the embryo, or never included dish/vignette
  "negative" examples), converting them to boxes will carry the same bias
  into the new detector. Garbage in, garbage out applies regardless of
  architecture choice — in particular, make sure training data includes
  frames where a dish rim or light vignette is visible, labeled with the
  correct (tight, embryo-only) box, so the detector explicitly learns to
  ignore that background structure rather than just never seeing it.

**Verdict:** A reasonable, likely net-positive path *if* pursued alongside
fixing/auditing the training labels — but it is not a guaranteed silver
bullet on its own, and it's more engineering effort than the near-zero-cost
fixes in Phase 1 below, which should be tried first.

### Option B: Train on more data with correct labeling (same architecture)

**Pros:** Generally the highest-ROI lever for any deep learning accuracy
problem, and complementary to any architecture choice — even if you
switch to YOLO, you'll still want good data. More diverse, correctly
labeled data (including negative/background examples of dish rims and
light vignettes) directly addresses root cause E and can partially
compensate for root cause C (architecture limitations).

**Cons / caveats:**
- The training pipeline and original dataset gap has been closed:
  `train_segmenter.py` now exists (dataset loader, Dice/BCE loss, train/val
  loop, checkpointing, fine-tuning from the production checkpoint), and
  the original ~30-pair paper dataset plus 10 newly labeled pairs are
  available (38 total — see Status table). The remaining cost is
  **labeling more, targeted data**, not building infrastructure from
  scratch.
- It does **not** fix root causes A/B (the union-of-all-contours and
  missing size/border sanity checks) at all — those are deterministic
  post-processing bugs that exist independent of model quality. More data
  won't help symptom 2 (dish/light treated as embryo) if the
  post-processing logic is the actual culprit.
- It cannot fix root cause D (source images that never captured the full
  embryo) — no amount of training data teaches a model to see pixels that
  were never recorded.
- "More data" is expensive relative to its marginal benefit if it's not
  targeted: **blindly collecting more of the same kind of image** you
  already have plenty of has diminishing returns. **Targeted** collection
  of the specific cases the current model already fails on (via the
  active-learning approach in Phase 3 below) is far more sample-efficient
  than bulk collection.
- Pixel-mask labeling (polygon/brush annotation) is significantly more
  time-consuming per image than bounding-box labeling, given §1's finding
  that only a box is actually needed downstream — this is a strong
  argument for labeling *new* hard examples as boxes (cheaper) rather than
  full masks, regardless of which model architecture ultimately consumes
  them.

**Verdict:** Necessary in the medium/long term almost no matter what, but
should be **targeted** (hard-example/active-learning driven) rather than
"just collect more data in general," and should not be the *first* thing
tried — it's the most expensive option, won't fix the union-of-contours
bug (symptom 2) on its own, and can't fix genuinely clipped source images
(symptom 3 / root cause D) at all.

### Option C: Other approaches worth considering

Roughly ordered from cheapest/fastest to most involved:

0. **Fix the shape-distortion bug (root cause F) — near-zero engineering
   cost, no retraining required, arguably the single highest-visual-impact
   fix available.** Square up the bounding box before cropping/resizing:
   expand the shorter side to match the longer side, centered on the
   original box, clamped to stay within the image, *then* crop and resize
   to 224×224 (now a uniform, non-distorting scale). This is already
   implemented and validated against real production images in
   `run_segmenter_inspect.py` (`squarify_bbox()`, `_6_cropped_224_fixed.png`
   output) — porting the same ~15 lines into `preprocessing_images.py`'s
   `process_image()` is a small, low-risk change. Note this doesn't
   replace fix #1 below (rejecting bad contours) — an asymmetric,
   border-touching box will still pull in extra background once
   squared, rather than recovering missing embryo detail; do both.
1. **Fix the contour-selection logic (root causes A + B) — near-zero
   engineering cost, no retraining required, should be tried first:**
   **Partially implemented in `run_segmenter_inspect.py` (`select_best_contour()`).**
   - Instead of unioning *every* contour that clears a circularity
     threshold, **keep only the single best candidate**. The implemented
     version **does not use circularity at all** (reported only as a
     diagnostic) — it ranks by area (largest wins) with a tie-break toward
     the frame center, after preferring contours that pass border/size
     sanity checks. Circularity was dropped deliberately: hatching and
     multi-cell embryos are often non-circular, while dish rims are
     border-touching and oversized regardless of shape.
   - **Explicitly reject contours that touch or nearly touch the image
     border**, and/or whose area exceeds some fraction (default 90%) of
     the full frame — a dish rim or light vignette will almost always
     trip one of these checks, while a real (even off-center) embryo
     usually won't. If no contour passes, fall back to the full candidate
     pool (so a genuinely edge-framed embryo is not vetoed outright).
   - Flag final crops covering > 70% of the frame as
     `possible_background_false_positive` (implemented).
   - Apply morphological closing (`cv2.morphologyEx` with `MORPH_CLOSE`)
     to merge small mask fragments before contour-finding, which helps
     with irregular/hatching shapes (symptom 1) without needing a shape
     heuristic at all.
   - Pad the final bounding box by a fixed number of pixels (or a % of
     its size) before cropping, as cheap insurance against slight
     under-segmentation (helps symptom 3's model-recall component).
   - Distinguish "no valid candidate found, used full frame defensively"
     from "used full frame because the best candidate covered almost the
     whole frame anyway" in your logging (see Phase 0) — right now both
     look identical downstream but have very different causes.
2. **Always combine original + brightness-corrected results**, rather than
   only falling back to the brightness-corrected pass when the first pass
   finds nothing. Taking the best-single-candidate (per #1) from the union
   of both attempts' contours (rather than either/or) could recover
   embryo area that's only visible in one of the two versions.
3. **Add skip connections to `SegmenterModel`** (turn it into a real
   U-Net, keeping its current lightweight conv blocks). **Done** —
   implemented as `train_segmenter.py --skip-connections`
   (`SegmenterModelSkip`, defined in that file; the production
   `SegmenterModel.py` is untouched, so nothing changes unless this flag
   is explicitly used). Measured cost: **+28% CPU inference time (+9
   ms/image), +12.5% parameters** (see performance doc §3.1); only 5
   conv-weight tensors differ in shape from the production checkpoint, so
   fine-tuning still starts from mostly-pretrained weights, not from
   scratch. **Caveat found by actually testing it:** with only ~38 labeled
   examples and `--freeze-encoder`, the 5 newly-shaped decoder layers
   start randomly initialized and, at least on one held-out image, this
   produced a visibly noisier raw mask than the fully-pretrained
   production decoder — which, combined with the noise-speck bug fixed
   under Root cause B, made the fine-tuned skip model look *worse* on
   that image even though its raw mask shape was comparable. This isn't
   evidence against skip connections; it means the new capacity needs
   more data/epochs (or an unfrozen encoder) to actually converge before
   a fair comparison against production is possible — re-run
   `run_segmenter_inspect.py` (now with the noise-speck fix) on a larger
   held-out set after further training. A further, separate option —
   replacing the encoder with a *real* Inception-module backbone
   (optionally ImageNet-pretrained), matching what the paper's
   "InceptionV3" name implies (see §3, item #1) — would cost far more
   (~24M params / ~5.7 GFLOPs vs. ~3.1M / ~1.2 GFLOPs, an ~8×/~5× jump)
   and risks overfitting even more given the same ~38 examples; only
   worth it with substantially more data or heavy layer-freezing.
4. **Use a pretrained/foundation segmentation model** (e.g. Meta's
   Segment Anything Model, SAM/SAM2) prompted with a rough
   center-point/box, either as the production model (if latency budget
   allows) or as a **labeling accelerator** — use it to quickly generate
   high-quality mask proposals for human review, dramatically speeding up
   Option B's data collection cost.
5. **Temporal consistency.** These are time-lapse video frames — the
   embryo doesn't teleport between consecutive frames, and neither does
   the dish/vignette (it's static). Smoothing the predicted bounding box
   (e.g. a rolling median over a small window of neighboring frames within
   the same embryo folder) would cheaply suppress single-frame failures
   of all three kinds without touching the model at all, and a bounding
   box that's implausibly large/unstable relative to neighboring frames is
   itself a good signal that symptom 2 just occurred on that frame.
6. **Active learning / hard-example mining.** Instrument the current
   pipeline (see Phase 0) to flag images where the fallback path
   triggers, where the chosen box covers almost the entire frame
   (symptom 2 proxy), or where the box touches the frame border (symptom 3
   proxy). These flagged images are exactly the highest-value examples to
   manually review/re-label and add to future training data — far more
   efficient than random bulk data collection (ties back into Option B).
7. **Classical CV hybrid.** Since the zona pellucida is roughly circular,
   `cv2.HoughCircles` on the raw (or lightly denoised) image is a cheap,
   dependency-light classical baseline that could run as an ensemble
   member alongside the CNN mask (e.g. take the union or intersection of
   both), adding robustness essentially for free in compute terms. Note it
   would need the same border/size sanity checks as #1, or it will be
   just as prone to "detecting" the dish rim as a circle.

## 5. Recommended Phased Roadmap

| Phase | Effort | What | Why first/next |
|---|---|---|---|
| **0. Diagnose** | Hours | Instrument `process_image` to log/save, per image: (a) fraction of frame area covered by the final crop; (b) whether the final box touches any image border; (c) how often the brightness-retry and full-image fallbacks trigger; (d) mask/bbox visualizations for manual review. | **Largely done** in `run_segmenter_inspect.py` (3 diagnostic PNGs per image + console metrics, with flagged images copied into `needs_review/` for manual review). Still worth porting the same logging into `preprocessing_images.py` for batch runs. |
| **1. Quick wins** | Days | **(a)** Square up the bounding box before cropping/resizing (root cause F). **(b)** Replace circularity-union with single best-contour selection (no circularity; border/size rejection + area/centrality scoring). **(c)** Morphological closing. **(d)** Bounding-box safety margin. | **(a) and (b) done in inspect tool**, validated on 7 problematic images (see Status table above). **(c) and (d) not yet.** Next: port (a)+(b) into `preprocessing_images.py`, then re-run diagnostics at scale. Zero retraining required. |
| **2. Architecture** | Days–1 week | **Done, opt-in** (`train_segmenter.py --skip-connections`, +28% inference cost, +12.5% params — see performance doc §3.1), reusing most of the current checkpoint. Remaining work: train with enough data/epochs (or an unfrozen encoder) for the new layers to actually converge, then re-validate with `run_segmenter_inspect.py` on a larger held-out set — see the caveat under Root cause C / Option C item 3. | Cheapest architecture fix for root cause C; directly addresses the paper's "U-Net" gap. |
| **2b. Bigger architecture bet (optional, defer)** | 1–3 weeks | Either (a) a real Inception-module backbone (optionally ImageNet-pretrained — a much bigger jump, ~8×/~5× more params/FLOPs, needs careful freezing given current data scale), or (b) converting existing masks to boxes and training a lightweight detector (YOLO-nano class) instead — informed by Phase 2's results and Phase 0's diagnostics. | Only pursue once Phase 2's cheaper fix has been tried and/or the labeled dataset has grown meaningfully. |
| **3. Targeted data expansion** | Ongoing | Use Phase 0's hard-example flags (plus any new failures found after Phase 1/2) to drive **targeted** new data collection/labeling — ideally as cheap bounding boxes rather than full masks (per §1), explicitly including "hard negative" frames with a visible dish rim/vignette so the model learns to ignore them, optionally accelerated by a prompted foundation model (SAM) to speed up annotation. Separately, audit and flag/exclude source images where the embryo appears genuinely clipped at acquisition time (root cause D) — this is a data-quality fix, not a model-training fix. Retrain periodically as new hard examples accumulate. | Highest long-term payoff, but should be targeted at known failure modes rather than done blindly, and is naturally an ongoing process rather than a one-time fix. |
| **4. Re-validate efficiency** | Days | Once a new model is chosen, re-run the efficiency analysis from `SEGMENTER_PERFORMANCE_ANALYSIS.md` (GPU batching, thread settings, etc.) against the new model to confirm the constant-factor recommendations still apply. | Efficiency work is largely orthogonal to the model architecture (per that doc) — worth confirming rather than assuming after a model swap. |

## 6. Direct Answers to the Questions Asked

- **"If I replace the CNN with a YOLO model, will it successfully increase
  accuracy and still be efficient?"** — Efficiency: very likely yes, a
  nano-scale detector is roughly the same compute/parameter budget as the
  current model, and arguably cheaper since a detection head is lighter
  than a full pixel-decoder (which, per §1, is discarded anyway).
  Accuracy: **plausibly yes for a real chunk of the problem** — specifically
  symptoms 1 and 2, since a detector has no circularity assumption and no
  "union every candidate contour" logic, so it structurally can't
  reproduce root causes A/B. It is **not guaranteed** to fix symptom 3 on
  its own, since that depends on model recall (root cause C, fixable with
  better architecture/data) and/or genuinely incomplete source images
  (root cause D, not fixable by *any* model). Note a training pipeline
  now exists (`train_segmenter.py`) but it's for the current mask-based
  CNN, not a detector — adopting YOLO still means building a separate
  pipeline.
- **"Or should we just train on more data with correct labeling?"** — Yes,
  this should happen eventually under any architecture, but **not as the
  first step**, and not as *undirected* bulk collection — target it at the
  specific failure cases surfaced by Phase 0's diagnostics (including
  explicit "dish rim / light vignette" hard negatives, since the model may
  never have been taught to reject them) for much better return on
  labeling effort. Take advantage of the fact that only bounding boxes
  (not full masks) are actually needed (§1), which makes new labeling
  significantly cheaper than it might first appear. Note that more data
  alone won't fix symptom 2 if the union-of-contours post-processing bug
  (root cause A) is the dominant cause, and can't fix symptom 3 cases
  caused by genuinely clipped source images (root cause D).
- **"Are there other approaches you suggest?"** — Yes: before touching the
  model at all, fix the contour-selection logic (Phase 1 — reject
  border-touching/oversized contours, pick a single best candidate instead
  of unioning all passers, square up the crop, add a bounding-box margin).
  The first two items are **already prototyped and validated** in
  `run_segmenter_inspect.py`; porting them to production is the immediate
  next step. Beyond that, adding skip connections (U-Net-style),
  leveraging a pretrained foundation segmentation model (SAM) to
  accelerate labeling, temporal smoothing across video frames, and
  active-learning-driven data collection targeting all three symptom
  categories are all worth considering, as detailed in §4.
