import numpy as np
from PIL import Image
import os
import torch
from EmbryoSegmenter.SegmenterModel import SegmenterModelSkip
import cv2
import concurrent.futures
import time

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

weights_dir = 'EmbryoSegmenter/weights_skip/'
weights_file = weights_dir+'latest.pt'
model_ft = SegmenterModelSkip()
model_ft.load_state_dict(torch.load(weights_file, map_location='cpu'))
model_ft.eval()

print("Segmentation Model loaded.")

def segment_image(frame):
    frame = frame/255.0
    # if frame has rgb, just get the first channel
    if len(frame.shape) > 2:
        frame = frame[:, :, 0]
    temp_frame = frame
    select_frames = temp_frame[None, None, ...]
    select_frames = np.array(select_frames)
    batch = torch.tensor(select_frames)
    batch = batch.float()
    with torch.no_grad():
        outputs = model_ft(batch)
    masks = outputs.detach().numpy()
    image = select_frames[0, 0, :, :]
    mask = masks[0, 0, :, :]
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
        # No positive pixels at all: fall back to the full image bounds.
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find the indices of rows and columns where the mask is 1
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    return (min_x, min_y, max_x, max_y)

def squarify_bbox(min_x, min_y, max_x, max_y, img_w, img_h):
    # Pads the box to a square before cropping, so the later resize to
    # 224x224 doesn't stretch the embryo.
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
    # Keeps one contour: prefer ones not touching the border and not oversized
    # (dish rim, vignette), then the largest, ties broken by closeness to center.
    cv2mask = (mask * 255).astype(np.uint8)
    h, w = cv2mask.shape[:2]
    contours, _ = cv2.findContours(cv2mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cv2mask)

    frame_area = float(w * h)
    frame_cx, frame_cy = w / 2.0, h / 2.0

    all_candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
    if not all_candidates:
        return (final_mask / 255.0).astype(float), None

    # Discard specks smaller than min_area_frac before anything else.
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

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

images_dir = '<path to images>'
cropped_images_dir = '<path of where to save processed images>'
images = os.listdir(images_dir)
total_images = len(images)
print("Total images:", total_images)


# Make a parallelized version of the above code
def process_image(image):
    img = load_image(images_dir+image)
    if np.max(img) > 0:
        # Always brightness-correct first, then segment once (no retry).
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

start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    # list() so an exception in any worker is raised here instead of silently dropped.
    results = list(executor.map(process_image, images))
end = time.time()
print("Time taken:", end-start)
