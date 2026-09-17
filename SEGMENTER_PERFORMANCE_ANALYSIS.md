# Embryo Segmenter: Complexity, Performance, and Accuracy Analysis

This document analyzes the asymptotic time complexity of the embryo
segmentation/preprocessing pipeline
(`preprocessing/preprocessing_images.py` +
`preprocessing/EmbryoSegmenter/SegmenterModel.py`), and evaluates whether
it needs significant algorithmic improvement, or whether gains should
instead come from implementation-level (constant-factor) changes.

**Looking to improve accuracy instead (e.g. the segmenter missing part of
the embryo)?** See
[`SEGMENTER_ACCURACY_ROADMAP.md`](SEGMENTER_ACCURACY_ROADMAP.md), which
covers root-cause diagnosis and evaluates YOLO vs. more training data vs.
other fixes.

**Bottom line up front:** The pipeline is already asymptotically optimal
— O(N) in the number of images, which is the best possible since every
image must be examined at least once. There is no Big-O improvement
available. However, there are several concrete, practical inefficiencies
(CPU-only execution, unbatched single-image inference, a
multiprocessing-portability bug, and duplicated logic) that are cheap to
fix and could yield large **constant-factor** (real wall-clock) speedups,
plus a couple of low-risk accuracy improvements worth considering.

**Update (September 2026):** Phase 1 accuracy fixes have been prototyped
in `preprocessing/run_segmenter_inspect.py` (see
[`SEGMENTER_ACCURACY_ROADMAP.md`](SEGMENTER_ACCURACY_ROADMAP.md) Status
section). §6 below compares the **old** production post-processing
(circularity filter + union of all passing contours) against the **new**
logic (`select_best_contour` + optional `squarify_bbox`). The CNN forward
pass is unchanged there; §3.1 separately quantifies the cost of a
*different*, not-yet-implemented change — adding skip connections to
`SegmenterModel` itself.

## 1. Pipeline Overview

For each of the N input images, `process_image()` in
`preprocessing_images.py` does:

1. `load_image` — decode the image file and resize to a fixed 224x224.
2. `segment_image` — one forward pass through `SegmenterModel` (a small
   PyTorch CNN) producing a 224x224 mask; threshold at 0.5.
3. `cv2.findContours` + contour filtering — in **production**
   (`preprocessing_images.py`): keep every contour whose circularity
   exceeds 0.7 and union them all; in the **prototyped fix**
   (`run_segmenter_inspect.py`): pick a single best candidate by
   border/size sanity checks + area/centrality (no circularity filter).
4. **Fallback path (only triggered if step 3 finds nothing):** re-run
   steps 2–3 on a brightness/contrast-corrected copy of the image. If that
   also finds nothing, skip cropping entirely (use the full frame).
5. `find_bounding_box` + crop + resize back to 224x224 + save.

All N images are processed independently via
`concurrent.futures.ProcessPoolExecutor`.

## 2. Asymptotic Time Complexity

### Per-image cost

Every step in `process_image()` operates on data whose size is **fixed**
(224x224), because `load_image` resizes immediately:

| Step | Depends on original image size? | Depends on 224x224 fixed size? |
|---|---|---|
| File decode (`Image.open`) | Yes — O(P), P = original pixel count | — |
| Resize to 224x224 | O(P) (resampling reads every source pixel once) | — |
| CNN forward pass (`segment_image`) | No | O(1) — fixed-size network, see §3 |
| Mask threshold | No | O(1) (224×224 = 50,176 fixed elements) |
| `cv2.findContours` | No | O(1) (bounded by 224×224 mask) |
| Contour filtering (circularity-union *or* best-candidate selection) | No | O(1) (bounded by mask size; see §6) |
| `squarify_bbox` (new, optional) | No | O(1) |
| Bounding box / crop / resize / save | No | O(1) |

So the per-image cost is **O(P)** where P is the original image's pixel
count (dominated by file decode/resize), plus a **fixed constant** C for
the CNN + mask post-processing (independent of P). In the worst case
(segmentation fails on the first attempt), steps 2–3 run twice, which at
most **doubles** the constant C — it does not change the asymptotic order.

### Overall complexity

Across N images: **O(N · (P̄ + C))**, where P̄ is the average original
image resolution and C is the fixed per-image model/post-processing cost.
Since decoding/reading each image at least once is unavoidable, and C is
a fixed constant independent of N, **this is already linear and
asymptotically optimal** — you cannot process N images in better than
O(N) time in general, and this pipeline achieves exactly that. There is
no algorithmic ("Big-O") restructuring that would meaningfully improve
this.

Parallelizing across processes (`ProcessPoolExecutor`, defaulting to
`os.cpu_count()` workers) does not change this asymptotic complexity — it
only reduces the **wall-clock constant** by roughly a factor of the number
of available CPU cores, since the images are embarrassingly parallel
(fully independent).

## 3. How Expensive Is the Constant C? (FLOP Analysis of SegmenterModel)

`SegmenterModel` is a small 6-layer encoder / 6-layer decoder CNN with a
fixed 224x224x1 input. Counting multiply-accumulate operations (MACs) for
each 3x3 convolution (`MACs = kernel² × Cin × Cout × Hout × Wout`):

| Layer | Cin → Cout | Spatial size | MACs |
|---|---|---|---|
| conv1 | 1 → 16 | 224×224 | ~7.2M |
| conv2 | 16 → 32 | 112×112 | ~57.8M |
| conv3 | 32 → 64 | 56×56 | ~57.8M |
| conv4 | 64 → 128 | 28×28 | ~57.8M |
| conv5 | 128 → 256 | 14×14 | ~57.8M |
| conv6 | 256 → 512 | 7×7 | ~57.8M |
| deconv5 | 512 → 256 | 7×7 | ~57.8M |
| deconv4 | 256 → 128 | 14×14 | ~57.8M |
| deconv3 | 128 → 64 | 28×28 | ~57.8M |
| deconv2 | 64 → 32 | 56×56 | ~57.8M |
| deconv1 | 32 → 16 | 112×112 | ~57.8M |
| decoded | 16 → 1 | 224×224 | ~7.2M |
| **Total** | | | **~592M MACs (~1.2 GFLOPs)** |

Notably, every middle layer costs almost exactly the same number of MACs
(~57.8M), because the architecture halves spatial resolution while
doubling channel count at each step — a nicely balanced design. ~1.2
GFLOPs per image is a **very small** model by modern standards (for
reference, ResNet-50 is ~4 GFLOPs and MobileNetV2 is ~0.3 GFLOPs), so the
constant C is inherently small — the architecture itself is not the
bottleneck.

**Measured** in `preprocessing/.venv-segmenter` (CPU, single-threaded,
batch=1, 100-iteration average after warmup): **32.8 ms/image**, matching
the FLOP estimate well. On a GPU with batching, this drops to well under a
millisecond per image (not directly measured here — no GPU was available
in this environment — but batched GPU inference for a model this size is
routine).

### 3.1 Cost of adding skip connections (U-Net-style)

The paper describes this component as an "InceptionV3 **U-Net**", but
`SegmenterModel` has no skip connections between encoder and decoder (see
[`SEGMENTER_ACCURACY_ROADMAP.md`](SEGMENTER_ACCURACY_ROADMAP.md) §3 and
Root cause C for why this plausibly hurts mask boundary accuracy). This
section quantifies the cost of *just* adding skip connections to the
existing lightweight architecture (concatenating each encoder stage's
pre-pool feature map into the matching decoder stage) — not the larger,
separate question of using a real Inception-module backbone (see the
roadmap for that comparison, which is much more expensive).

Measured by implementing this variant and benchmarking both side by side
(same environment as above):

| | Current (no skip) | + Skip connections | Change |
|---|---|---|---|
| MACs | ~592M (~1.2 GFLOPs) | ~831M (~1.7 GFLOPs) | +40% (analytical) |
| Parameters | 3,145,954 | 3,537,778 | +12.5% (measured) |
| CPU forward pass (1 thread, batch=1) | 32.8 ms | 41.9 ms | **+28%** (measured) |

This is a real but small cost in absolute terms (+9 ms/image on CPU,
negligible once GPU-batched per §5). Only 4 of the 6 decoder layers change
shape (`deconv4`/`3`/`2`/`1` and `decoded`) — the encoder and `deconv5`
are untouched, so a checkpoint fine-tuned this way can reuse most of the
current production weights rather than retraining from scratch.

## 4. Where Real-World Time Is Likely Actually Going

Since the asymptotic complexity is already optimal and the model itself is
small, the **practical** bottlenecks are implementation details, not
algorithmic ones:

1. **Forced CPU execution, unbatched (batch size 1).** `segment_image`
   hardcodes `map_location='cpu'` / `model_ft.to("cpu")` and processes one
   image at a time (`select_frames[None, None, ...]` → batch dimension of
   1). Even though the model is small, running many tiny batch-size-1 CPU
   forward passes pays fixed per-call overhead (Python/PyTorch dispatch,
   kernel launch, cache effects) on every single image rather than
   amortizing it across a batch. This doesn't change the O(N) *order*, but
   can easily be a 5–50x constant-factor cost compared to GPU-batched
   inference.

2. **CPU-thread oversubscription risk.** PyTorch's own intra-op
   parallelism (via MKL/OpenMP threads) is enabled by default. When
   combined with `ProcessPoolExecutor`'s multi-process parallelism, you
   can end up with `(# processes) × (# PyTorch threads per process)`
   competing for the same physical cores, causing contention that
   *reduces* effective throughput rather than increasing it. This is a
   very common and easy-to-miss pitfall with PyTorch + multiprocessing.

3. **Multiprocessing portability/correctness risk.** The script's
   top-level code (model load, `os.listdir`, and the
   `ProcessPoolExecutor` call) is not guarded by `if __name__ ==
   "__main__":`. On Linux (default `fork` start method) this happens to
   work because child processes are cloned from the already-initialized
   parent. On platforms/Python configurations that default to `spawn`,
   each worker would re-execute the entire module from scratch — reloading
   the model and re-listing the directory once per worker (or worse,
   attempting to recursively spawn a new pool). This is a portability
   footgun worth fixing defensively even if you only ever run on Linux
   today.

4. **Duplicated segmentation/contour logic.** The "first pass" and
   "brightness-corrected retry" blocks in `process_image` are
   near-identical copy-pasted code. This isn't a performance issue, but it
   is a maintainability/correctness risk (a fix applied to one copy could
   easily be forgotten in the other).

5. **Double-cost fallback path.** Whenever the first segmentation attempt
   yields no acceptable contour, the image is fully reprocessed (segmented
   again + contours again). If this fallback triggers frequently on your
   dataset, it's worth instrumenting how often it happens — if it's
   common, that's more of an *accuracy* signal (the model/threshold isn't
   robust enough for a meaningful fraction of your images) than a
   performance one. On a 7-image problematic sample, the old
   circularity-union logic triggered this on 2/7 images; the new
   `select_best_contour` logic triggered it on 0/7 (see §6.3) — a case
   where an accuracy fix directly improves throughput too.

## 5. Recommendations

None of these change the Big-O complexity (which is already optimal) —
they target the constant factor and robustness:

**High impact, low risk:**
- **Batch inference on GPU.** Modify `segment_image` (or add a new
  batched variant) to accept a batch of images, move the model to GPU
  (`model_ft.to("cuda")`) when available, and process images in batches of
  e.g. 32–128 rather than one at a time. This is the single biggest
  likely speedup for large datasets and requires no architecture changes.
- **Guard top-level script code** with `if __name__ == "__main__":`
  (wrapping the model loading, directory listing, and executor call). This
  is a small, safe change that removes the fork/spawn portability risk
  entirely.
- **Cap PyTorch's intra-op thread count per worker** (e.g.
  `torch.set_num_threads(1)` inside each worker) when using
  `ProcessPoolExecutor`, so that process-level and thread-level
  parallelism don't compete for the same cores.

**Medium impact:**
- **Deduplicate the retry logic** in `process_image` into a small helper
  function (e.g. `segment_and_filter(img) -> final_mask`) called twice.
  Purely a maintainability win, but reduces the chance of the two copies
  drifting apart.
- **Track/log the fallback-trigger rate** (how often the brightness-retry
  path or the "give up, use full frame" path is hit) as a data-quality
  metric. A high rate would indicate the segmentation model itself needs
  retraining/improvement for accuracy, which batching/GPU work won't fix.

**Accuracy-focused (minimal performance cost — see §6):**
- **Port `select_best_contour` + `squarify_bbox`** from
  `run_segmenter_inspect.py` into `preprocessing_images.py`. Adds
  ~24 µs/image post-processing but can eliminate costly double forward
  passes on images where the old circularity filter finds nothing.

**Lower priority / accuracy-focused (not performance):**
- **Consider skip connections (U-Net-style)** in `SegmenterModel` — see
  §3.1 for the measured cost (+28% CPU inference time, +12.5% parameters;
  still just +9 ms/image). See the accuracy roadmap for the expected
  quality benefit and why this is cheaper than a full Inception-module
  backbone rewrite.
- **Revisit the fixed thresholds** (mask binarization at 0.5; in
  production, `min_circularity = 0.7`). The inspect-tool prototype has
  replaced the circularity threshold entirely with border/size rejection
  (`max_area_frac=0.9`, `border_margin=2`) — see §6 and the accuracy
  roadmap for rationale.

## 6. Old vs. New Post-Processing: Performance Comparison

This section compares the **production** contour logic in
`preprocessing_images.py` against the **Phase 1 prototype** in
`run_segmenter_inspect.py`. The neural network (`SegmenterModel`) is
identical in both paths — only the mask-to-bounding-box step differs.

### 6.1 What changed (accuracy side, briefly)

| Step | Old (production) | New (inspect tool prototype) |
|---|---|---|
| Contour filter | Union all contours with circularity > 0.7 | Single best contour: prefer non-border-touching, ≤90% frame area; score by area then centrality |
| Circularity | Used as hard filter | **Not used for selection** (reported as diagnostic only) |
| Crop resize | Non-square box → force 224×224 (distorts) | `_6` output: `squarify_bbox()` then uniform resize (no distortion) |
| Faulty-detection flag | None | `possible_background_false_positive` when crop > 70% of frame |

Validated on 7 problematic images: full-image fallback dropped from 1→0,
brightness fallback 1→0, >70% crop false-positives 1→0 (see accuracy
roadmap Status table for details).

### 6.2 Per-step micro-benchmark (post-processing only)

Measured on this machine (CPU, synthetic 224×224 masks with 0–5
contours, 30,000 total calls per variant):

| Variant | Time per image | vs. old |
|---|---|---|
| Old: circularity filter + union all passers | **436 µs** | baseline |
| New: `select_best_contour` | **454 µs** | +4% (~18 µs slower) |
| New: `select_best_contour` + `squarify_bbox` | **460 µs** | +6% (~24 µs slower) |
| `SegmenterModel` forward pass (CPU, batch=1) | **32.8 ms** | — |

**Interpretation:** The new logic adds a handful of extra OpenCV calls per
contour (`boundingRect`, `moments`, `hypot`) and a `max()` over
candidates instead of a list-comprehension + union draw. In absolute
terms this is **negligible**: post-processing is ~**1.4%** of per-image
CPU time (0.46 ms vs 32.8 ms), and the new path is only ~**0.06%** slower
end-to-end than the old one. **Big-O is unchanged** — still O(1) per
image, O(k) in the number of contours k where k is bounded by the fixed
224×224 mask (typically k ≪ 224).

The dominant cost remains the CNN forward pass (~70× more expensive than
even the slower new post-processing step). GPU batching recommendations
from §5 therefore still apply unchanged.

### 6.3 Fallback-path throughput (where the new logic can actually save time)

The old logic sometimes finds **no** circularity-passing contour, triggering:
1. A second full forward pass on a brightness-corrected copy, and/or
2. A full-frame fallback (`np.ones_like`).

Each triggered fallback **doubles** the CNN cost for that image (second
forward pass) or wastes crop quality (full frame). On the 7-image
validation set:

| Fallback | Old logic | New logic |
|---|---|---|
| Brightness retry (extra forward pass) | 1 / 7 (14%) | **0 / 7** |
| Full-image fallback | 1 / 7 (14%) | **0 / 7** |

If fallback rates on a larger dataset are similar, the new post-processing
can yield a **net throughput win** despite being ~4% slower on the
post-processing micro-benchmark alone: avoiding one extra 32.8 ms forward
pass per image saves far more than the ~18 µs added contour scoring costs.

**Rough break-even:** if the new logic avoids a second forward pass on
≥ 0.06% of images (18 µs / 32.8 ms), it pays for its own extra
post-processing overhead. On the 7-image sample, 14% of images avoided a
double pass — a large margin.

### 6.4 Memory and I/O (inspect tool only)

`run_segmenter_inspect.py` writes 6 PNG artifacts per image for manual
review. That extra disk I/O is **not** part of the production
post-processing cost model and should not be counted when estimating
batch-preprocessing throughput. Production `preprocessing_images.py`
would only pay the `select_best_contour` + `squarify_bbox` CPU cost if
ported.

**Bottom line: no new bottlenecks.** Porting `select_best_contour` +
`squarify_bbox` to production is safe from a performance perspective —
negligible added CPU cost (§6.2), and may reduce fallback-triggered double
inference (§6.3). GPU batching and multiprocessing hygiene (§5) remain the
primary speed levers regardless of which contour logic runs.

## 7. Conclusion

The segmenter pipeline's asymptotic complexity — O(1) per image, O(N)
overall — is already as good as it can be for a per-image, independent
preprocessing task; **no algorithmic rewrite is needed or would help**.
The opportunities for "improving efficiency" identified in this analysis
are all constant-factor engineering improvements (GPU batching being the
highest-leverage one).

The Phase 1 accuracy fixes prototyped in `run_segmenter_inspect.py` **do
not materially change that picture**: post-processing remains O(1) and
sub-millisecond, ~70× cheaper than inference, and the new contour logic
is within a few percent of the old step on a micro-benchmark while
potentially **reducing end-to-end time** on datasets where the old
circularity-union logic frequently triggers the double-pass fallback.
Port these fixes to production for the accuracy win; pursue GPU batching
and multiprocessing hygiene for the speed win.
