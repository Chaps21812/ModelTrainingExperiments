from astropy.io import fits
import numpy as np
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np


def _iqr_clip(x, threshold=5.0):
    """
    IQR-Clip normalization: Robust contrast normalization with hard clipping.
    
    Args:
        x (np.ndarray): Grayscale image, shape (H, W)
    
    Returns:
        np.ndarray: Normalized and clipped image, same shape, dtype float32
    """
    x = x.astype(np.float32)
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    # Normalize relative to the median (q2)
    x_norm = (x - q2) / (iqr + 1e-8)

    # Clip values beyond ±5 IQR
    x_clipped = np.clip(x_norm, -threshold, threshold)

    return x_clipped

def _iqr_log(x, threshold=5.0):
    """
    IQR-Log normalization: IQR-based normalization followed by log compression of outliers.
    
    Args:
        x (np.ndarray): Grayscale image, shape (H, W)
    
    Returns:
        np.ndarray: Soft-clipped image using log transform for values > ±5 IQR
    """
    x = x.astype(np.float32)
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    # Normalize relative to the median (q2)
    x_soft = (x - q2) / (iqr + 1e-8)

    # Apply log transformation to soft-clip tails
    threshold = 5.0

    # Positive tail
    over = x_soft > threshold
    x_soft[over] = threshold + np.log1p(x_soft[over] - threshold)

    # Negative tail
    under = x_soft < -threshold
    x_soft[under] = -threshold - np.log1p(-x_soft[under] - threshold)

    return x_soft

def _adaptive_iqr(fits_image:np.ndarray, bkg_subtract:bool=True, verbose:bool=False) -> np.ndarray:
    '''
    Performs Log1P contrast enhancement. Searches for the highest contrast image and enhances stars.
    Optionally can perform background subtraction as well

    Notes: Current configuration of the PSF model works for a scale of 4-5 arcmin
           sized image. Will make this more adjustable with calculations if needed.

    Input: The stacked frames to be processed for astrometric localization. Works
           best when background has already been corrected.

    Output: A numpy array of shape (2, N) where N is the number of stars extracted. 
    '''  

    if verbose:
        print("| Percentile | Contrast |")
        print("|------------|----------|")
    best_contrast_score = 0
    best_percentile = 0
    best_image = None
    percentiles=[]
    contrasts=[]

    for i in range(20):
        #Scans image to find optimal subtraction of median
        percentile = 90+0.5*i
        temp_image = fits_image-np.quantile(fits_image, (percentile)/100)
        temp_image[temp_image < 0] = 0
        scaled_data = np.log1p(temp_image)
        #Metric to optimize, currently it is prominence
        contrast = (np.max(scaled_data)+np.mean(scaled_data))/2-np.median(scaled_data)
        percentiles.append(percentile)
        contrasts.append(contrast)

        if contrast > best_contrast_score*1.05:
            best_contrast_multiplier = i
            best_image = scaled_data.copy()
            best_contrast_score = contrast
            best_percentile = percentile
        if verbose: print("|    {:.2f}   |   {:.2f}   |".format(percentile,contrast))
    if verbose: print("Best percentile): {}".format(best_percentile))
    if best_image is None:
        return fits_image
    return best_image

def _zscale(image:np.ndarray, contrast:float=.5) -> np.ndarray:
    scalar = ZScaleInterval(contrast=contrast)
    return scalar(image)

def _minmax_scale(arr:np.ndarray) -> np.ndarray:
    """Scales a 2D NumPy array to the range [0, 1] using min-max normalization."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=float)  # Avoid division by zero
    return (arr - arr_min) / (arr_max - arr_min)

def _background_subtract(data:np.ndarray) -> np.ndarray:
    W = data.shape[0]
    H = data.shape[1]

    kernel_size_x = int(W/16)
    kernel_size_y = int(H/16)
    if kernel_size_x %2 == 0: kernel_size_x +=1
    if kernel_size_y %2 == 0: kernel_size_y +=1

    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (int(W/4), int(H/4)), filter_size=(kernel_size_x, kernel_size_y),
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data = data-bkg.background
    return data

def _median_row_subtraction(img):
    """
    Subtracts the median from each row and adds back the global median.

    Args:
        img (np.ndarray): Input image of shape (H, W).

    Returns:
        np.ndarray: Processed image of shape (H, W), dtype float32.
    """
    img = img.astype(np.float32)
    global_median = np.median(img)
    row_medians = np.median(img, axis=1, keepdims=True)
    result = img - row_medians + global_median
    return result

def _median_column_subtraction(img):
    """
    Subtracts the median from each column and adds back the global median.

    Args:
        img (np.ndarray): Input image of shape (H, W).

    Returns:
        np.ndarray: Processed image of shape (H, W), dtype float32.
    """
    img = img.astype(np.float32)
    global_median = np.median(img)
    col_medians = np.median(img, axis=0, keepdims=True)
    result = img - col_medians + global_median
    return result

def adaptiveIQR(data:np.ndarray) -> np.ndarray:
    contrast_enhance = _adaptive_iqr(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    return np.stack([contrast_enhance, contrast_enhance, contrast_enhance], axis=0)

def zscale(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)

    return np.stack([zscaled, zscaled, zscaled], axis=0)

def iqr_clipped(data, threshold=5) -> np.ndarray:
    data = _iqr_clip(data, threshold)
    data = (_minmax_scale(data)*255).astype(np.uint8)
    return np.stack([data]*3, axis=0)

def iqr_log(data, threshold=5) -> np.ndarray:
    data = _iqr_log(data, threshold)
    data = (_minmax_scale(data)*255).astype(np.uint8)
    return np.stack([data]*3, axis=0)

def channel_mixture_A(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)
    contrast_enhance = _iqr_clip(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    data = (data / 65535*255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def channel_mixture_B(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)
    contrast_enhance = _adaptive_iqr(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    data = (data / 65535*255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def channel_mixture_C(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)
    contrast_enhance = _iqr_log(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    data = (data / 65535*255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def raw_file(data: np.ndarray) -> np.ndarray:
    return  np.stack([data/65535*255]*3, axis=0).astype(np.uint8)


def raw_file_16bit(data: np.ndarray) -> np.ndarray:
    return  np.stack([data], axis=0).astype(np.uint16)

def iqr_clipped_16bit(data, threshold=5) -> np.ndarray:
    data = _iqr_clip(data, threshold)
    data = (_minmax_scale(data)*65535).astype(np.uint16)
    return data

def iqr_log_16bit(data, threshold=5) -> np.ndarray:
    data = _iqr_log(data, threshold)
    data = (_minmax_scale(data)*65535).astype(np.uint16)
    return data



if __name__=="__main__":
        from data_connections.src.coco_tools import silt_to_coco, satsim_to_coco, merge_coco
        from preprocess_functions import channel_mixture, adaptiveIQR, zscale

        Process_pathE = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset-Statistics/data/PDS-RME04-2024-05-07"
        final_destination_path = "/mnt/c/Users/david.chaparro/Documents/data/ZScaledRME03AllStar"
        silt_to_coco(Process_pathE, include_sats=True, include_stars=False, convert_png=True, process_func=channel_mixture, notes="Channel Mixture of Adaptive IQR, raw, and zscaled. ")