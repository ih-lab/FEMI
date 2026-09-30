# --------------------------------------------------------------------------
# preprocessing_images.py
#
# This is the main script that preprocesses a directory of raw embryo time-lapse
# images before they are used to pretrain/fine-tune FEMI.
# Originally, for each image, this script would:
#   1. Loads and resizes the image to 224x224.
#   2. Runs it through a small PyTorch segmentation model (SegmenterModel)
#      to predict a mask of "which pixels belong to the embryo".
#   3. Cleans up that mask by keeping only sufficiently circular contours
#      (the embryo/zona pellucida is roughly circular, so this filters out
#      noise, dish edges, pipettes, etc.).
#   4. If segmentation completely fails (empty mask), retries once after
#      automatically correcting brightness/contrast.
#   5. Crops the image to the bounding box of the cleaned mask and resizes
#      back to 224x224, then saves it to the output directory.
#
# Now this script makes the following changes:
#   2. Segments with SegmenterModelSkip (a U-Net variant with skip
#      connections, trained via train_segmenter.py --skip-connections;
#      see EmbryoSegmenter/SegmenterModel.py) instead of the original
#      SegmenterModel, and always applies automatic brightness/contrast
#      correction first (one forward pass per image, no retry). 
#   4. Picks a single best contour via border/size rejection + area/centrality
#      scoring (`select_best_contour`) instead of unioning every sufficiently
#      circular contour.
#   5. If no valid contour is found, falls back to using the whole image.
#   6. Squares up the bounding box (`squarify_bbox`), crops, and uniformly
#      resizes to 224x224 (no aspect-ratio warping), then saves to the output
#      directory.
#
# Images are processed in parallel across CPU cores using
# `concurrent.futures.ProcessPoolExecutor` for speed, since each image is
# handled independently.
#
# NOTE (paths): `images_dir` and `cropped_images_dir` below are placeholders
# and must be edited before running this script directly.
#
# NOTE (portability): This script runs top-level code (model loading,
# directory listing, and the ProcessPoolExecutor call) at import time
# without an `if __name__ == "__main__":` guard. On Linux (which uses the
# "fork" multiprocessing start method by default) this works because worker
# processes are cloned from the already-initialized parent process. On
# platforms that default to the "spawn" start method (Windows/macOS on
# newer Python), each worker process would re-import this file from
# scratch, which would reload the model, re-list the directory, and
# potentially recursively re-launch the executor. See README/analysis docs
# for more detail on this and other performance considerations.
# --------------------------------------------------------------------------

import numpy as np
from PIL import Image
import os
import torch
from EmbryoSegmenter.SegmenterModel import SegmenterModelSkip
import cv2
import concurrent.futures
import time

def load_image(image_path):
    """Load an image from disk and resize it to the model's fixed 224x224
    input resolution. Returns a numpy array (H, W) or (H, W, C) depending on
    the source image's mode."""
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

# ---------------------------------------------------------------------
# Load the pretrained segmentation model once, at import time, so it can
# be reused across every image processed by this script/process. The model
# is forced onto the CPU (map_location='cpu', model_ft.to("cpu")) and put
# into eval() mode to disable dropout/batchnorm training behavior and
# gradient tracking overhead is avoided later via .detach() calls.
# ---------------------------------------------------------------------
weights_dir = 'EmbryoSegmenter/weights_skip/'
weights_file = weights_dir+'latest.pt'
model_ft = SegmenterModelSkip()
model_ft.load_state_dict(torch.load(weights_file, map_location='cpu'))
model_ft = model_ft.to("cpu")
model_ft.eval()

print("Segmentation Model loaded.")

def segment_image(frame):
    """
    Run the segmentation model on a single frame and return both the
    masked/segmented image and the raw binary mask.

    Parameters:
    - frame: a 2D (grayscale) or 3D (e.g. RGB) numpy array, already resized
      to 224x224, with pixel values in [0, 255].

    Returns:
    - segmented_images: the input image with all non-embryo pixels zeroed
      out (mask applied), scaled back to [0, 255] and cast to int.
    - mask: the raw binary (0/1) mask (float array) produced by thresholding
      the model's sigmoid output at 0.5.

    Complexity: this performs exactly one forward pass through
    SegmenterModelSkip on a fixed 224x224x1 input, so its cost is O(1)
    (constant) regardless of the original image's resolution.
    """
    frame = frame/255.0
    # if frame has rgb, just get the first channel
    if len(frame.shape) > 2:
        frame = frame[:, :, 0]
    temp_frame = frame
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W), since the
    # model expects a batch of single-channel images.
    select_frames = temp_frame[None, None, ...]
    select_frames = np.array(select_frames)
    batch = torch.tensor(select_frames)
    batch = batch.float()
    # Forward pass through the encoder/decoder network (see
    # SegmenterModelSkip.forward). No torch.no_grad() context is used here, but
    # .detach() below strips gradients before converting to numpy, so no
    # gradients are actually retained/backpropagated.
    outputs = model_ft(batch)
    masks = outputs.detach().numpy()
    image = select_frames[0, 0, :, :]
    mask = masks[0, 0, :, :]
    # Binarize the sigmoid probability mask at a 0.5 threshold.
    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1
    res = image * mask * 255
    segmented_images = np.array(res).astype(int)
    return segmented_images, mask

def find_bounding_box(mask):
    """
    Find the bounding box of a mask where the mask is a 2D array of 1s and 0s.
    
    Parameters:
    - mask: A 2D numpy array of 1s and 0s.
    
    Returns:
    A tuple of (min_x, min_y, max_x, max_y) representing the bounding box.
    """
    # Find the rows and columns where the mask is 1
    if np.sum(mask) == 0:
        # No positive pixels at all: fall back to the full image bounds
        # rather than an undefined/empty box.
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find the indices of rows and columns where the mask is 1
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    return (min_x, min_y, max_x, max_y)

def squarify_bbox(min_x, min_y, max_x, max_y, img_w, img_h):
    """
    Expand a (possibly non-square) bounding box into a square box of side
    max(width, height), centered on the original box, then shift it so it
    stays inside the image. Cropping a square first makes the later resize
    to 224x224 uniform, so the embryo is not stretched.
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

def select_best_contour(mask, max_area_frac=0.9, border_margin=2, min_area_frac=0.1):
    """
    Pick a single best-candidate contour instead of unioning every contour
    whose circularity clears a threshold.

    Discards specks smaller than `min_area_frac` of the frame, prefers
    contours that do not touch the border and are not larger than
    `max_area_frac` of the frame, and if none pass those checks falls back
    to the full candidate pool. The largest remaining contour wins, with
    ties broken by proximity to the frame center.

    Returns (final_mask, selected_contour). final_mask is float 0.0/1.0.
    selected_contour is None if no contour was found.
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

def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    """
    Automatically rescale an image's brightness/contrast by clipping a
    percentage of the darkest/lightest pixels (based on the cumulative
    grayscale histogram) and linearly stretching the remaining range to
    fill [0, 255]. Applied to every frame before segmentation, so dark or
    washed-out images are corrected before the single forward pass.
    """
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    if (maximum_gray - minimum_gray) == 0:
        return image
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    # alpha (contrast gain) and beta (brightness offset) are applied via
    # cv2's convertScaleAbs: new_pixel = |alpha * pixel + beta|, clipped to
    # [0, 255] and cast back to uint8.
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

# ---------------------------------------------------------------------
# Script configuration: EDIT THESE PATHS before running.
# `images_dir` should contain the raw source images; processed/cropped
# images will be written to `cropped_images_dir` using the same filenames.
# ---------------------------------------------------------------------
images_dir = '<path to images>'
cropped_images_dir = '<path of where to save processed images>'
images = os.listdir(images_dir)
total_images = len(images)
print("Total images:", total_images)


def process_image(image):
    """
    Full per-image preprocessing pipeline, designed to be called once per
    image (independently of all other images), which makes it a natural
    unit of work to distribute across processes.

    Steps:
      1. Load + resize to 224x224.
      2. Skip fully-blank images (np.max(img) == 0).
      3. Always brightness-correct, then segment once.
      4. select_best_contour (single best candidate; no circularity filter).
      5. If no valid contour, fall back to the full frame.
      6. Squarify the bounding box, crop, uniformly resize to 224x224, save.

    Complexity: every step operates on a fixed 224x224 image, so per-image
    cost is O(1) aside from the initial file decode.
    """
    img = load_image(images_dir+image)
    if np.max(img) > 0:
        # Always correct brightness/contrast before the single forward pass.
        bright_img = automatic_brightness_and_contrast(img)
        segmented_images, mask = segment_image(bright_img)
        final_mask, _selected_contour = select_best_contour(mask)
        if np.max(final_mask) == 0:
            # No valid contour: keep the full frame so the image isn't discarded.
            final_mask = np.ones_like(final_mask)
        min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
        h, w = img.shape[0], img.shape[1]
        min_x, min_y, max_x, max_y = squarify_bbox(min_x, min_y, max_x, max_y, w, h)
        cropped_image = img[min_y:max_y, min_x:max_x]
        cropped_image = Image.fromarray(cropped_image)
        cropped_image = cropped_image.resize((224, 224))
        cropped_image.save(cropped_images_dir+image)
    return image

# ---------------------------------------------------------------------
# Fan out `process_image` across a pool of worker processes (defaults to
# os.cpu_count() workers). Since each image is handled fully independently,
# this is an "embarrassingly parallel" workload: wall-clock time scales
# down roughly linearly with the number of CPU cores available, though the
# total AMOUNT of work done (and therefore the asymptotic complexity in
# terms of number of images) is unchanged - see analysis docs.
# ---------------------------------------------------------------------
start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(process_image, images)
end = time.time()
print("Time taken:", end-start)
