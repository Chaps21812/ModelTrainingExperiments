from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import mlflow
from tqdm import tqdm
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from .format_targets import format_targets_bboxes

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
    return results
        
