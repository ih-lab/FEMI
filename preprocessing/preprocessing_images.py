# --------------------------------------------------------------------------
# preprocessing_images.py
#
# Standalone script that preprocesses a directory of raw embryo time-lapse
# images before they are used to pretrain/fine-tune FEMI. For each image,
# this script:
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
from EmbryoSegmenter.SegmenterModel import SegmenterModel
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
weights_dir = 'EmbryoSegmenter/weights/'
weights_file = weights_dir+'epoch_102_loss_0.098710.pt'
model_ft = SegmenterModel()
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
    SegmenterModel on a fixed 224x224x1 input, so its cost is O(1)
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
    # SegmenterModel.forward). No torch.no_grad() context is used here, but
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

def calculate_circularity(contour):
    """
    Compute how close a contour's shape is to a perfect circle using the
    isoperimetric-style formula: circularity = 4*pi*Area / Perimeter^2.
    A perfect circle scores 1.0; more irregular/elongated shapes score
    lower. Used to filter out non-embryo blobs (e.g. dish artifacts) from
    the predicted mask, since embryos are roughly circular/spherical.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter ** 2))

def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    """
    Automatically rescale an image's brightness/contrast by clipping a
    percentage of the darkest/lightest pixels (based on the cumulative
    grayscale histogram) and linearly stretching the remaining range to
    fill [0, 255]. This is used as a fallback preprocessing step when the
    segmentation model fails to find any embryo-like region on the raw
    image (e.g. because the frame is too dark/washed out).
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
    clip_hist_percent *= (maximum/100.0)
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


# Make a parallelized version of the above code
def process_image(image):
    """
    Full per-image preprocessing pipeline, designed to be called once per
    image (independently of all other images), which makes it a natural
    unit of work to distribute across processes.

    Steps:
      1. Load + resize to 224x224.
      2. Skip fully-blank images (np.max(img) == 0).
      3. Segment -> binarize -> keep only sufficiently circular contours.
      4. If that produced an empty mask, auto-correct brightness/contrast
         and retry the segmentation + circularity filtering once.
      5. If it's STILL empty after the retry, fall back to using the whole
         image (no crop) rather than producing an unusable empty crop.
      6. Crop to the final mask's bounding box, resize back to 224x224, and
         save to `cropped_images_dir`.

    Complexity: every step here operates on a fixed 224x224-sized image, so
    the per-image cost is O(1) (constant), aside from the initial file
    decode which scales with the ORIGINAL image's resolution. In the worst
    case (segmentation fails twice), the constant-factor cost roughly
    doubles because segment_image() and the contour/circularity pipeline
    run twice.
    """
    img = load_image(images_dir+image)
    if np.max(img) > 0:
        # Segment the image
        segmented_images, mask = segment_image(img)
        cv2mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter the contours based on circularity
        min_circularity = 0.7  # This is a threshold for circularity
        circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
        # Create an empty mask for the final output
        final_mask = np.zeros_like(cv2mask)
        cv2.drawContours(final_mask, circular_contours, -1, (255), thickness=cv2.FILLED)
        final_mask = final_mask / 255.0
        final_mask = final_mask.astype(float)
        if np.max(final_mask) == 0:
            # First-pass segmentation found nothing sufficiently circular.
            # Retry once after normalizing brightness/contrast, which often
            # rescues frames that were too dark/washed out for the model.
            bright_img = automatic_brightness_and_contrast(img)
            segmented_images, mask = segment_image(bright_img)
            cv2mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter the contours based on circularity
            min_circularity = 0.7
            circular_contours = [cnt for cnt in contours if calculate_circularity(cnt) > min_circularity]
            # Create an empty mask for the final output
            final_mask = np.zeros_like(cv2mask)
            cv2.drawContours(final_mask, circular_contours, -1, (255), thickness=cv2.FILLED)
            final_mask = final_mask / 255.0
            final_mask = final_mask.astype(float)
            if np.sum(final_mask) == 0:
                # Still nothing found even after the brightness retry: give
                # up on cropping and just keep the full frame so the image
                # isn't discarded entirely.
                final_mask = np.ones_like(final_mask)
        min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
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
