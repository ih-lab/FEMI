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

## 1. Pipeline Overview

For each of the N input images, `process_image()` in
`preprocessing_images.py` does:

1. `load_image` — decode the image file and resize to a fixed 224x224.
2. `segment_image` — one forward pass through `SegmenterModel` (a small
   PyTorch CNN) producing a 224x224 mask; threshold at 0.5.
3. `cv2.findContours` + `calculate_circularity` — find contours in the
   mask and keep only those that look sufficiently circular (embryos are
   roughly round).
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
| `calculate_circularity` per contour | No | O(1) (perimeter ≤ O(224+224)) |
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
bottleneck. On a modern CPU with optimized BLAS/MKL-DNN kernels (which
PyTorch uses by default), this should take on the order of single-digit
to tens of milliseconds per image; on a GPU, well under a millisecond when
batched.

*(Note: no GPU/PyTorch benchmarking environment was available while
writing this analysis — the sandbox used has 1 CPU core and no PyTorch
installed — so the above is an analytical FLOP-based estimate rather than
a measured benchmark. We recommend running a quick timing test on your
actual hardware, e.g. `python -c "import time, torch; from
EmbryoSegmenter.SegmenterModel import SegmenterModel; m =
SegmenterModel().eval(); x = torch.rand(1,1,224,224); [m(x) for _ in
range(3)]; t=time.time(); [m(x) for _ in range(100)];
print((time.time()-t)/100)"` to get a real per-image latency number on
your machine.)*

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
   yields no sufficiently circular contour, the image is fully
   reprocessed (segmented again + contours again). If this fallback
   triggers frequently on your dataset, it's worth instrumenting how often
   it happens — if it's common, that's more of an *accuracy* signal
   (the model/threshold isn't robust enough for a meaningful fraction of
   your images) than a performance one.

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

**Lower priority / accuracy-focused (not performance):**
- **Consider skip connections (U-Net-style)** between encoder and decoder
  layers in `SegmenterModel`. The current architecture has none, so all
  boundary detail must be reconstructed from a 7×7×512 bottleneck; skip
  connections typically improve mask boundary precision substantially in
  segmentation networks, at a modest compute cost (still well under 1
  GFLOP total).
- **Revisit the fixed thresholds** (mask binarization at 0.5,
  `min_circularity = 0.7`). These are reasonable heuristics but were
  presumably tuned on a specific dataset; if your images differ
  meaningfully (different scope/lighting/magnification), it may be worth
  re-validating them or making them CLI-configurable rather than hardcoded.

## 6. Conclusion

The segmenter pipeline's asymptotic complexity — O(1) per image, O(N)
overall — is already as good as it can be for a per-image, independent
preprocessing task; **no algorithmic rewrite is needed or would help**.
The opportunities for "improving efficiency" identified in this analysis
are all constant-factor engineering improvements (GPU batching being the
highest-leverage one), and the opportunities for "improving accuracy" are
architectural/heuristic-tuning questions (skip connections, threshold
tuning, monitoring the fallback rate) rather than complexity questions.
