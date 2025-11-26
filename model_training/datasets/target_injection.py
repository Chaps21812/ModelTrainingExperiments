import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from scipy.ndimage import rotate
import random
import json
from pathlib import Path


def extract_segmented_patches_from_json_and_fits(
    json_path:str, 
    fits_path:str,
    threshold_factor=1.2, 
    bbox_scale=1.5
):
    """
    Extracts and segments patches from a FITS image using annotations in a JSON file.

    Parameters:
        json_path (str or Path): Path to the annotation JSON file.
        threshold_factor (float): Multiplier of global median to threshold target.
        bbox_scale (float): Factor to scale the bounding box.

    Returns:
        List of dicts: {
            'patch': 2D NumPy array (segmented),
            'original_patch': 2D array (raw),
            'bbox': [x, y, w, h],
            'image_id': image ID
        }
    """
    json_path = Path(json_path)
    folder = json_path.parent

    with open(json_path, "r") as f:
        data = json.load(f)

    image_info = data["images"][0]
    fits_path = folder / image_info["file_name"]
    image_id = image_info["id"]

    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data

    h_img, w_img = img_data.shape

    patches = []

    for ann in data["annotations"]:
        if ann["image_id"] != image_id:
            continue

        # Original bbox
        x, y, w, h = ann["bbox"]

        # Expand bbox
        cx = x + w / 2
        cy = y + h / 2
        new_w = w * bbox_scale
        new_h = h * bbox_scale
        x0 = int(max(cx - new_w / 2, 0))
        y0 = int(max(cy - new_h / 2, 0))
        x1 = int(min(cx + new_w / 2, w_img))
        y1 = int(min(cy + new_h / 2, h_img))

        original_patch = img_data[y0:y1, x0:x1]
        median = np.median(original_patch)
        threshold = threshold_factor * median

        mean_pixel = np.mean(original_patch)
        std_pixel = np.std(original_patch)

        segmented_patch = np.where(original_patch >= mean_pixel+std_pixel, original_patch, 0)

        patches.append({
            "patch": segmented_patch,
            "original_patch": original_patch,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "image_id": image_id
        })

    return patches

def inject_target_into_fits(
    image:np.ndarray,
    patch:np.ndarray,
    random_rotation=True,
    display=True,
    seed=None,
    max_radius=None, 
    min_radius=0  
):
    """
    Injects a segmented target patch into a random location in the FITS image.

    Parameters:
        image (np.ndarray): Path to FITS file.
        patch (np.ndarray): 2D segmented patch (zeros outside the target).
        random_rotation (bool): Whether to randomly rotate the target.
        display (bool): If True, plot the image with zscale and target bbox.
        seed (int or None): Random seed for reproducibility.

    Returns:
        Tuple (injected_image, injection_bbox), where:
            - injected_image: 2D NumPy array with target added
            - injection_bbox: [x0, y0, w, h] of injected region
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_data = image.copy()  # Don't modify in-place

    h_img, w_img = img_data.shape
    h_patch, w_patch = patch.shape

    if random_rotation:
        angle = random.choice([0, 90, 180, 270])
        patch = rotate(patch, angle, reshape=False, order=1, mode='constant', cval=0)

    # Ensure target fits in image
    cx, cy = w_img // 2, h_img // 2

    # Precompute valid region to avoid patch going out of bounds
    x_min = w_patch // 2
    x_max = w_img - w_patch // 2
    y_min = h_patch // 2
    y_max = h_img - h_patch // 2

    # Set radius limits
    max_r = max_radius if max_radius is not None else min(cx, cy)
    min_r = min_radius

    for _ in range(100):  
        r = np.random.uniform(min_r, max_r)
        theta = np.random.uniform(0, 2 * np.pi)
        dx = int(r * np.cos(theta))
        dy = int(r * np.sin(theta))

        x0 = cx + dx - w_patch // 2
        y0 = cy + dy - h_patch // 2

        if x0 >= 0 and y0 >= 0 and x0 + w_patch <= w_img and y0 + h_patch <= h_img:
            break
    else:
        raise RuntimeError("Failed to find valid injection point within radius constraints.")

    # Inject (only overwrite where patch is nonzero)
    #Examine injection code here
    patch_mask = patch > 0
    img_data[y0:y0 + h_patch, x0:x0 + w_patch][patch_mask] += patch[patch_mask].astype(np.uint16)

    injection_bbox = [x0/w_img, y0/h_img, w_patch/w_img, h_patch/h_img]

    if display:
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(img_data)
        plt.figure(figsize=(16, 12))
        plt.imshow(img_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        rect = plt.Rectangle(
            (x0, y0), w_patch, h_patch,
            edgecolor='red', facecolor='none', linewidth=1
        )
        plt.gca().add_patch(rect)
        plt.title(f"Injected Target at ({x0}, {y0}) with size ({w_patch}x{h_patch})")
        plt.axis("off")
        plt.show()

    return img_data, injection_bbox
