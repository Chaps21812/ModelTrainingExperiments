# from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
# import torch
# import torchvision
# from torchvision.datasets import CocoDetection
# import torchvision.transforms.v2 as T
# from torch.utils.data import DataLoader
# import os
# import mlflow
# from tqdm import tqdm
# from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference
# from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
# from evaluation.dataset_studies import Dataset_study
# from training_frameworks.format_targets import format_targets_bboxes

# if __name__ == "__main__":

#     def evaluate_over_time(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
#         total_targets= []
#         total_predictions = []
#         metrics = {}

#         model.eval()
#         for images, targets in tqdm(dataloader, desc="Evaluating"):
#             images = list(img for img in images)
#             # images = list(img.to(device) for img in images)
#             targets = format_targets_bboxes(targets)
#             # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]




#             model = Sentinel()
#             model.eval()
#             model.load_original_model("/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/models/LMNT01_MixtureC/retinanet_weights_E249.pt")
#             TS_Model = torch.jit.script(model)
#             path = os.path.join("/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/models/LMNT01_MixtureC/","LMNT01-249-TS.torchscript" )
#             TS_Model.save(path)
#             print("torchvision version:", torchvision.__version__)

#             model = torch.jit.load(str(path))  # cast to string if pathlib.Path
#             model.eval()  # Ensure it's in inference mode


#             outputs = model(images)
#             total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
#             total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

#             if epoch%10 == 0 and plot:
#                 plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
#                 plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
#                 # plot=False

#         for metric in evaluation_metrics:
#             results = metric(total_predictions, total_targets)
#             metrics.update(results)
#         return metrics

#     def perform_timewise_benchmark(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5):
#         database = Dataset_study(output_directory, datasets, experiment_title)
#         # Custom transforms (RetinaNet expects images and targets)
#         device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
#         transform = T.Compose([
#             T.ToTensor(),
#             # T.RandomHorizontalFlip(0.5),
#             # T.Resize(1024),
#         ])
#         # Load model
#         model = Sentinel()
#         model.load_original_model("/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/models/LMNT01_MixtureC/retinanet_weights_E249.pt")
#         model.to(device)

#         mlflow.set_tracking_uri("http://localhost:5000")
#         mlflow.set_experiment(experiment_title)
#         mlflow.log_param("Model", model_path)
#         mlflow.log_param("Dataset", datasets)
#         mlflow.end_run()
#         with mlflow.start_run():
#             for index, path in tqdm(enumerate(database)):
#                 # Load COCO-style dataset
#                 validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
#                 validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True, collate_fn=lambda x: (zip(*x)))
#                 results = evaluate_over_time(model, path, index,validation_loader, evaluation_metrics, device)
#                 mlflow.log_metrics(results, index)
#                 results["date"] = database.path_to_date[path]
#                 database.add_metric(results)
#                 database.save()

#                 # Optimizer
#             database.plot_all_metrics()
#             mlflow.end_run()

#     LMNT01_Datasets = ["/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-13_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-25_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-15_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-23_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-15_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-26_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-17_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-20_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-23_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-03_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-16_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-25_Channel_Mixture_C_Eval"]

#     LMNT02_Datasets = ["/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-14_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-08_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-31_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-14_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-19_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-24_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-31_Channel_Mixture_C_Eval"]
    
#     RME04_Datasets = ["/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-07-19_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-09_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-04_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-11_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-13_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-08_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-06_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-07_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2025-06-04_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-15_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-28_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-12_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-05_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-07-09_Channel_Mixture_C"]

#     LMNT01model = "/data/Sentinel_Datasets/Best_models/LMNT01_Base_model.pt"
#     LMNT02model = "/data/Sentinel_Datasets/Best_models/LMNT02_Base_model.pt"
#     RME04model = "/data/Sentinel_Datasets/Best_models/RME04_Base_model.pt"
#     title = "TEsting"
#     output_dir = os.path.join("/data/Sentinel_Datasets/Experiments", title)
#     metrics = [centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference]

#     perform_timewise_benchmark(LMNT02_Datasets, LMNT01model, output_dir, title, metrics, device_no=3)


from typing import Any, cast
import pydantic
import torch
import base64
import io
import numpy as np
from fastapi.responses import StreamingResponse
from astropy.io import fits
from importlib import resources
import json
import os

import torchvision
from numpy import typing as npt
import numpy as np
import pathlib

import numpy as np
from astropy.visualization import ZScaleInterval
import numpy as np
from numpy import typing as npt
import cv2

def inference( data: list) -> list:
    path = os.path.join("/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/models/LMNT01_MixtureC/","LMNT01-249-TS.torchscript" )
    model = torch.jit.load(str(path))  # cast to string if pathlib.Path
    model.eval()  # Ensure it's in inference mode

    batch_detections: list[dict[str, list[dict[str, Any]]]] = [
        {"detections": []} for _ in data
    ]
    sidereal_detections = 0
    images = []
    rate_indices = []
    x_resolutions = []
    y_resolutions = []

    for i, file in enumerate(data):
        decoded = base64.b64decode(file["file"])
        tempfits = fits.open(io.BytesIO(decoded))
        fitfile = tempfits[0]
        header = fitfile.header
        img_data = fitfile.data
        y_resolutions.append(img_data.shape[0])
        x_resolutions.append(img_data.shape[1])
        if header["TRKMODE"] == "sidereal":
            sidereal_detections += 1
            continue

        arr_float = channel_mixture_C(img_data)  # Expects [C, H, W] float32
        images.append(arr_float)
        rate_indices.append(i)

    if not images:
        return pydantic.TypeAdapter(list[entities.ObjectDetections]).validate_python(batch_detections)

    images_np = np.stack(images, axis=0)  # Shape: [B, C, H, W]
    batch = torch.from_numpy(images_np)

    with torch.no_grad():
        outputs = model(batch)  # Should be shape: [B, 5, N]

    for k, (orig_i, preds) in enumerate(zip(rate_indices, outputs)):
        detections = []
        preds = preds.permute(1, 0)  # Now shape: [N, 5]
        H, W = images_np[k].shape[1:]  # assume [C, H, W]

        for det in preds:
            x_c, y_c, w, h, conf = det.tolist()
            if conf < 0.5:  # Confidence threshold (adjustable)
                continue

            # Convert normalized center/size to pixel coordinates
            x_c *= W
            y_c *= H
            w *= W
            h *= H

            xmin = x_c - w / 2
            xmax = x_c + w / 2
            ymin = y_c - h / 2
            ymax = y_c + h / 2

            # Clamp to image bounds
            xmin = max(0, xmin)
            xmax = min(W - 1, xmax)
            ymin = max(0, ymin)
            ymax = min(H - 1, ymax)

            signal = images_np[k, 0, int(y_c), int(x_c)]
            noise = np.std(images_np[k, 0])

            detection = {
                "class_id": 0,  # Only one class
                "pixel_centroid": [float(x_c)/x_resolutions[k], float(y_c)/y_resolutions[k]],
                "prob": float(conf),
                "snr": float(signal / noise) if noise > 0 else 0,
                "x_max": float(xmax)/x_resolutions[k],
                "x_min": float(xmin)/x_resolutions[k],
                "y_max": float(ymax)/y_resolutions[k],
                "y_min": float(ymin)/y_resolutions[k],
            }
            detections.append(detection)

        batch_detections[orig_i]["detections"] = detections

    return batch_detections



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

    data = (data / 255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def channel_mixture_B(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled * 255).astype(np.uint8)
    contrast_enhance = _adaptive_iqr(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)*255).astype(np.uint8)

    data = (data / 255).astype(np.uint8)
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def channel_mixture_C(data:np.ndarray) -> np.ndarray:
    zscaled = _zscale(data)
    zscaled = (zscaled).astype(np.float32)
    contrast_enhance = _iqr_log(data)
    contrast_enhance = (_minmax_scale(contrast_enhance)).astype(np.float32)

    data = (data).astype(np.float32)/65535
    return np.stack([data, contrast_enhance, zscaled], axis=0)

def raw_file(data: np.ndarray) -> np.ndarray:
    return  np.stack([data/65535]*3, axis=0)

def preprocess_image( image: npt.NDArray) -> npt.NDArray:
    # Apply zscale to the image data for contrast enhancement
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    # Apply Z-scale normalization (clipping values between vmin and vmax)
    #image = np.clip(image, vmin, vmax)
    #image = (image - vmin) / (vmax - vmin) * 255  # Scale to 0-255 range
    # Convert the image data to an unsigned 8-bit integer (for saving as PNG)
    
    image = image.astype(np.float32)/65535.0

    height, width = image.shape
    new_height = (
        (height // 32) * 32 if height % 32 == 0 else ((height // 32) + 1) * 32
    )
    new_width = (width // 32) * 32 if width % 32 == 0 else ((width // 32) + 1) * 32
    #resized_image = cv2.resize(image, (new_width, new_height))
    resized_image = cv2.resize(image, (512, 512))
    image = np.stack([resized_image] * 3, axis=0)
    
    return image

# Path to your JSON file
file_path = "/home/davidchaparro/Repos/ModelTrainingExperiments/model_training/detect-with-metadata-fc46157d-13b9-461b-ac71-7a528f419e43.json"

# Load JSON data
with open(file_path, "r") as f:
    data = json.load(f)

inference(data)

