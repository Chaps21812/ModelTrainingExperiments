from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import mlflow
from tqdm import tqdm
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from .format_targets import format_targets_bboxes
from .image_stitching import partition_images, recombine_annotations, generate_crops
import models.LETR.src.util.misc as utils
import torch
import numpy as np


def evaluate(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
    total_targets= []
    total_predictions = []
    metrics = {}

    model.eval()
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

        if plot:
            plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics

def evaluate_stitching(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False, sub_batch_size=40):
    total_targets= []
    total_predictions = []
    metrics = {}

    model.eval()
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        images_cropped_predicted, empty_targets = generate_crops(images, device=device)

        sub_batch_image = [images_cropped_predicted[i:i + sub_batch_size] for i in range(0, len(images_cropped_predicted), sub_batch_size)]

        temporary_outputs = []
        with torch.no_grad():
            for sub_images in sub_batch_image:
                outputs = model(sub_images)
                temporary_outputs.extend(outputs)
                del sub_images
                torch.cuda.empty_cache() 

        recombined_targets_predicted = recombine_annotations(empty_targets, temporary_outputs, device=device)

        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in recombined_targets_predicted])

        if plot:
            plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics
        
@torch.no_grad()
def evaluate_LETR(model, criterion, postprocessors, dataloader, evaluation_metrics, device, plot:bool=False):
    total_targets= []
    total_predictions = []
    metrics = {}
    model.eval()
    criterion.eval()

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, origin_indices = model(images, postprocessors, targets, criterion)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['line'](outputs, orig_target_sizes, "prediction")
    
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in results])

        pred_logits = outputs['pred_logits']
        bz = pred_logits.shape[0]
        assert bz ==1 
        query = pred_logits.shape[1]

        rst = results[0]['lines']
        pred_lines = rst.view(query, 2, 2)

        pred_lines = pred_lines.flip([-1]) # this is yxyx format

        h, w = targets[0]['orig_size'].tolist()
        pred_lines[:,:,0] = pred_lines[:,:,0]*(128)   
        pred_lines[:,:,0] = pred_lines[:,:,0]/h
        pred_lines[:,:,1] = pred_lines[:,:,1]*(128)
        pred_lines[:,:,1] = pred_lines[:,:,1]/w
        
        score = results[0]['scores'].cpu().numpy()
        line = pred_lines.cpu().numpy()

        score_idx = np.argsort(-score)
        line = line[score_idx]
        score = score[score_idx]

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['line'](outputs, orig_target_sizes, "prediction")
        

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics



@torch.no_grad()
def evaluate_DEF_DETR(model, criterion, postprocessors, dataloader, evaluation_metrics, device):
    total_targets= []
    total_predictions = []
    metrics = {}
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))


    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in results])

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics