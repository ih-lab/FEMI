# Embryo Segmenter: Accuracy Improvement Roadmap

**Audience:** whoever picks up the "improve the embryo segmenter" task next.
**Companion doc:** see
[`SEGMENTER_PERFORMANCE_ANALYSIS.md`](SEGMENTER_PERFORMANCE_ANALYSIS.md)
for the efficiency/complexity analysis this roadmap builds on.

**Reported problems** (three distinct symptom categories observed in
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

### Root cause C — the model's raw mask recall is genuinely incomplete

Even before contour filtering, the network may under-predict the embryo's
true extent (contributes most directly to **Symptom 3**, bottom-of-embryo
missing). Plausible contributing factors given the architecture (see
`SegmenterModel.py`):
- **No skip connections** between encoder and decoder (unlike a classic
  U-Net) — all boundary detail must be reconstructed from a 7×7×512
  bottleneck, which is known to hurt fine boundary precision and can cause
  systematic under-prediction at edges (plausibly worse at the
  bottom/edges of frame if there's any positional bias in lighting or
  training data).
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
| 1. Hatching/irregular embryo missed | A (irregular contour scores < 0.7, excluded) | Phase 1: replace hard circularity threshold with a softer/largest-component selection |
| 2. Dish/light circle treated as embryo, minimal cropping | A + B (oversized/border-touching circular contour unioned in) | Phase 1: reject border-touching & oversized contours; pick single best candidate, don't union all |
| 3. Bottom of embryo missing | C (model under-recall) and/or D (source image already incomplete) | Phase 0 to distinguish which; Phase 1 margin as a band-aid; Phase 2 skip connections for C; data-quality audit/exclusion for D |
| 4. Final crop looks visibly stretched/warped (embryo shape distorted) | **F (non-square crop force-resized to a square) — confirmed present in production, affects ~43% of a first sample batch** | Phase 1: square up the bounding box (pad to `max(w,h)`, centered, clamped to image bounds) *before* resizing — implemented and validated in `run_segmenter_inspect.py`'s `squarify_bbox()` / `_6_cropped_224_fixed.png` output, ready to port into `preprocessing_images.py` |

## 3. Evaluating the proposed options

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
- **No training pipeline exists in this repo today** for the segmenter at
  all (only the model definition + inference script). Adopting YOLO does
  not avoid this — you'd need to build a training pipeline either way
  (see §4), likely using the `ultralytics` package for YOLO specifically,
  which is a new dependency.
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
- **This repo has no existing segmenter training script or dataset** — one
  would need to be built from scratch regardless (dataset loader, loss
  function [e.g. Dice/BCE], train/val loop, checkpointing).
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
   - Instead of unioning *every* contour that clears the circularity
     threshold, score each candidate and **keep only the single best
     one** — e.g. rank by a combination of circularity, area (within a
     plausible expected-embryo-size range for a 224×224 frame, tunable
     from a sample of good crops), and distance of the contour's centroid
     from the image center (embryos are typically framed near-center).
   - **Explicitly reject contours that touch or nearly touch the image
     border**, and/or whose area exceeds some fraction (e.g. 80–90%) of
     the full frame — a dish rim or light vignette will almost always
     trip one of these checks, while a real (even off-center) embryo
     usually won't.
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
   U-Net). Cheap in compute (still well under 1 GFLOP total) and directly
   targets root cause C's boundary-precision limitation, without
   needing a new training pipeline built around a different framework
   (still plain PyTorch). This requires retraining with the existing data
   format, so it's a good complement to Option B.
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

## 4. Recommended Phased Roadmap

| Phase | Effort | What | Why first/next |
|---|---|---|---|
| **0. Diagnose** | Hours | Instrument `process_image` to log/save, per image: (a) fraction of frame area covered by the final crop (flags symptom 2 — dish/vignette — when it's e.g. >85%); (b) whether the final box touches any image border (flags symptom 3 candidates — either model under-recall or a genuinely clipped source frame); (c) how often the brightness-retry fallback triggers, and how often the full-image fallback (`np.ones_like`) triggers; (d) a random sample of mask visualizations overlaid on the original image for manual review, stratified across the three symptom categories. | You need to know the actual prevalence of each root cause (A–E) before investing in retraining — e.g. the >85%-area-covered metric directly measures how common the union-of-contours bug (symptom 2) is. This costs almost nothing and de-risks every subsequent phase. |
| **1. Quick wins** | Days | **(a)** Square up the bounding box before cropping/resizing (root cause F — fixes symptom 4, already prototyped in `run_segmenter_inspect.py`); **(b)** replace "union of all circularity-passing contours" with "single best candidate" scored by circularity + plausible size range + centrality; reject border-touching/oversized contours outright (fixes symptom 2 and helps symptom 1); **(c)** apply morphological closing before contour-finding (helps symptom 1); **(d)** add a bounding-box safety margin (helps symptom 3's model-recall component). Re-run Phase 0's diagnostics to measure improvement on each symptom category separately. | Zero retraining required; directly fixes the most likely and cheapest-to-fix root causes (A/B/F — all post-processing logic, not the model). Should be done regardless of which long-term path you choose, and (a) in particular has an immediately visible, easy-to-demo impact. |
| **2. Architecture + data pipeline** | 1–3 weeks | Build the (currently-missing) training pipeline for this component. Decide between (a) adding skip connections to the existing CNN and continuing to train it on mask data, or (b) converting existing masks to boxes and training a lightweight detector (YOLO-nano class or a simple custom box-regression head) instead — informed by Phase 0's findings. Either way, reuse existing labeled masks; no new labels needed yet. | This is where the YOLO-vs-more-data decision actually gets made, backed by evidence from Phase 0 rather than guesswork. |
| **3. Targeted data expansion** | Ongoing | Use Phase 0's hard-example flags (plus any new failures found after Phase 1/2) to drive **targeted** new data collection/labeling — ideally as cheap bounding boxes rather than full masks (per §1), explicitly including "hard negative" frames with a visible dish rim/vignette so the model learns to ignore them, optionally accelerated by a prompted foundation model (SAM) to speed up annotation. Separately, audit and flag/exclude source images where the embryo appears genuinely clipped at acquisition time (root cause D) — this is a data-quality fix, not a model-training fix. Retrain periodically as new hard examples accumulate. | Highest long-term payoff, but should be targeted at known failure modes rather than done blindly, and is naturally an ongoing process rather than a one-time fix. |
| **4. Re-validate efficiency** | Days | Once a new model is chosen, re-run the efficiency analysis from `SEGMENTER_PERFORMANCE_ANALYSIS.md` (GPU batching, thread settings, etc.) against the new model to confirm the constant-factor recommendations still apply. | Efficiency work is largely orthogonal to the model architecture (per that doc) — worth confirming rather than assuming after a model swap. |

## 5. Direct Answers to the Questions Asked

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
  (root cause D, not fixable by *any* model). Also note there's currently
  **no training pipeline for this component at all**, so adopting YOLO
  requires building one regardless.
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
  of unioning all passers, add a bounding-box margin) — this is nearly
  free, requires no retraining, and should meaningfully reduce symptoms 1
  and 2 by itself. Beyond that, adding skip connections (U-Net-style),
  leveraging a pretrained foundation segmentation model (SAM) to
  accelerate labeling, temporal smoothing across video frames, and
  active-learning-driven data collection targeting all three symptom
  categories are all worth considering, as detailed in §3.
