import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import os
import mlflow
from tqdm import tqdm
from evaluation.evaluation_metrics import centroid_l2_accuracy_Per_Image
from evaluation.dataset_studies import Dataset_study
from training_frameworks.format_targets import format_targets_bboxes
import torch.nn.functional as F
from typing import Union



def evaluate_model(model, dataloader:DataLoader, evaluation_metrics:list, device:str, image_attrs:list=["num_sats"], annot_attrs:list=["local_snr"], plot:bool=False):
    total_targets= []
    total_predictions = []
    max_prediction_length = 0
    max_target_length = 0
    metrics = {}
    H=1
    W=1

    image_attributes = {}
    for att in annot_attrs:
        image_attributes[att] = []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)
            H=images[0].shape[1]
            W=images[0].shape[2]
                    
            for att in annot_attrs:
                image_attributes.extend([sum(annot[att] for annot in target)/len(target) if len(target) > 0 else 0.0 for target in targets])
            targets = format_targets_bboxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            tensor_targets = model.output_formatter.convert_targets(targets)

            max_prediction_length = max(max_prediction_length, outputs.shape[2])
            max_target_length = max(max_target_length, tensor_targets.shape[2])
            total_predictions.append(outputs)
            total_targets.append(tensor_targets)

            # total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
            # total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

            # if epoch%10 == 0 and plot:
            #     plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            #     plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
            #     # plot=False

    padded_predictions = [F.pad(t, pad=(0, max_target_length - t.shape[2])) for t in total_predictions]
    padded_targets = [F.pad(t, pad=(0, max_prediction_length - t.shape[2])) for t in total_targets]

    total_preds = torch.cat(padded_predictions, dim=0)
    total_tgts = torch.cat(padded_targets, dim=0)

    concurrent_tps = centroid_l2_accuracy_Per_Image(total_preds, total_tgts)
    
    metrics.update(concurrent_tps)
    for key,value in image_attributes.items():
        metrics[key] = value

    for metric in evaluation_metrics:
        results = metric(total_preds, total_tgts, image_height=(W,H))
        metrics.update(results)
    return metrics

def perform_timewise_benchmark(datasets:list, model:torch.Module, output_directory:str, experiment_title, evaluation_metrics, batch_size:int = 20, tracked_attributes:list=["local_snr"], device_no=5, database=None):
    if database is None:
        database = Dataset_study(output_directory, datasets, experiment_title)
    else:
        database = database
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.ToTensor()])
    model.eval()

    for index, path in tqdm(enumerate(database)):
        # Load COCO-style dataset
        validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: (zip(*x)))
        results = evaluate_model(model,validation_loader, evaluation_metrics, device, annot_attrs=tracked_attributes)
        results["date"] = database.path_to_date[path]
        database.add_metric(results)
        database.save()
        mlflow.log_metrics(results, index)

        # Optimizer
    database.plot_all_metrics()
    return database

def perform_cumulative_evaluation(datasets:Union[list,str], model:torch.Module, output_directory:str, experiment_title, evaluation_metrics, batch_size = 20, tracked_attributes:list=["local_snr"], device_no=5, database=None):
    if isinstance(datasets,str):
        datasets = [datasets]
    if database is None:
        database = Dataset_study(output_directory, datasets, experiment_title)
    else:
        database = database
    # Custom transforms (RetinaNet expects images and targets)
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.ToTensor()])
    model.eval()

    # Load COCO-style dataset
    validation_sets = []
    for validation_dir in datasets:
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_sets.append(temp_validation)

    validation_set = ConcatDataset(validation_sets)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: (zip(*x)))

    results = evaluate_model(model,validation_loader, evaluation_metrics, device, annot_attrs=tracked_attributes)
    database.cumulative_metrics = results
    database.save()
    mlflow.log_params(results)

    return database
