import numpy as np
from PIL import Image
import os
import torch
from EmbryoSegmenter.SegmenterModel import SegmenterModel
import cv2
import concurrent.futures
import time

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

weights_dir = 'EmbryoSegmenter/weights/'
weights_file = weights_dir+'epoch_102_loss_0.098710.pt'
model_ft = SegmenterModel()
model_ft.load_state_dict(torch.load(weights_file, map_location='cpu'))
model_ft = model_ft.to("cpu")
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
        return (0, 0, mask.shape[1], mask.shape[0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find the indices of rows and columns where the mask is 1
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    return (min_x, min_y, max_x, max_y)

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter ** 2))

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
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

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
                final_mask = np.ones_like(final_mask)
        min_x, min_y, max_x, max_y = find_bounding_box(final_mask)
        cropped_image = img[min_y:max_y, min_x:max_x]    
        cropped_image = Image.fromarray(cropped_image)
        cropped_image = cropped_image.resize((224, 224))
        cropped_image.save(cropped_images_dir+image)
    return image

start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(process_image, images)
end = time.time()
print("Time taken:", end-start)