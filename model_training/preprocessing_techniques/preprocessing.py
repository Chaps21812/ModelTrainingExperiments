from astropy.io import fits
import numpy as np
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np

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

    if bkg_subtract:
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(fits_image, (32, 32), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        fits_image = fits_image-bkg.background

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

def adaptiveIQR(data:np.ndarray) -> np.ndarray:

    contrast_enhance = _adaptive_iqr(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    return np.stack([contrast_enhance, contrast_enhance, contrast_enhance], axis=0)

def channel_mixture(data:np.ndarray) -> np.ndarray:

    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)
    contrast_enhance = _adaptive_iqr(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    data = (data / 255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def zscale(data:np.ndarray) -> np.ndarray:

    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)

    return np.stack([zscaled, zscaled, zscaled], axis=0)


if __name__=="__main__":
    print("bruh")