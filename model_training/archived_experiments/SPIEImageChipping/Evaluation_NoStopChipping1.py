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
from typing import Union
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from models.Sentinel_Models.Sentinel_Retina_Net_NMS import SentinelNMS
from models.Sentinel_Models.Sentinel_Retina_Net_Stitch import Sentinel_Panoptic
from evaluation.evaluation_metrics import calculate_pr_curves, centroid_l2_accuracy, centroid_l2_accuracy_Per_Image
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from evaluation.dataset_studies import Dataset_study, Single_Dataset_Study
from training_frameworks.format_targets import format_targets_bboxes
import torch.nn.functional as F

def evaluate_single_dataset(model, dataloader:DataLoader, device:str, plot:bool=False) -> dict:
    total_targets= []
    total_predictions = []
    image_ids = []
    original_tgt_attributes = []
    max_prediction_length = 0
    max_target_length = 0
    metrics = {}

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)

            original_tgt_attributes.extend(targets)
            targets = format_targets_bboxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            tensor_targets = model.output_formatter.convert_targets(targets)

            image_ids.extend([t["image_id"].item() for t in targets])
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

    metrics["original_tgt_attributes"] = original_tgt_attributes

    padded_predictions = [F.pad(t, pad=(0, max_target_length - t.shape[2])) for t in total_predictions]
    padded_targets = [F.pad(t, pad=(0, max_prediction_length - t.shape[2])) for t in total_targets]
    total_preds = torch.cat(padded_predictions, dim=0)
    total_tgts = torch.cat(padded_targets, dim=0)

    concurrent_tps = centroid_l2_accuracy_Per_Image(total_preds, total_tgts)
    pr_curves = calculate_pr_curves(total_predictions, total_targets)
    metrics.update(concurrent_tps)
    metrics.update(pr_curves)

    confidence_thresholds = [.1, .5, .9]
    fit_thresholds = [.5,1]
    for tc in confidence_thresholds:
        for tf in fit_thresholds:
            results = centroid_l2_accuracy(total_predictions, total_targets, tconfidence=tc, tfit=tf)
            metrics.update(results)
    return metrics, total_preds, total_tgts, image_ids

def evaluate_on_test_sets(datasets:Union[str,list], model, output_directory:str, experiment_title, evaluation_metrics, device_no:torch.device, database=None):
    if isinstance(datasets, str):
        datasets = [datasets]
    if database is None:
        database = Single_Dataset_Study(output_directory, datasets, experiment_title)
    else:
        database = database

    transform = T.Compose([
        T.ToTensor(),
    ])

    # Load COCO-style dataset
    validation_sets = []
    for validation_dir in datasets:
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_sets.append(temp_validation)
    validation_set = ConcatDataset(validation_sets)
    validation_loader = DataLoader(validation_set, batch_size=20, shuffle=True, collate_fn=lambda x: (zip(*x)))

    results, predictions, gts, image_ids = evaluate_single_dataset(model,validation_loader, device)
    database.add_metrics(results)
    database.add_gts(gts)
    database.add_predictions(predictions)
    database.add_image_ids(image_ids)
    database.save()

    only_params = {k: v for k, v in results.items() if isinstance(v, float)}
    mlflow.log_params(only_params)
    return database

if __name__ == "__main__":
    #Pull data from these variables
    L1_10Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_10_Overlap/models/L1_10_Overlap_2025-08-15_15:43/L1_10_Overlap_weights_E75.pt"
    L1_20Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_20_Overlap/models/L1_20_Overlap_2025-08-15_15:43/L1_20_Overlap_weights_E75.pt"
    L1_30SmallOverlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_30_Overlap/models/NoStop_L1_30_SmallOverlap_2025-08-26_19:17/NoStop_L1_30_SmallOverlap_weights_E75.pt"
    L1_30Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_30_Overlap/models/NoStop_L1_30_Overlap_2025-08-26_06:57/NoStop_L1_30_Overlap_weights_E75.pt"
    L1_Random_Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_random_Overlap/models/L1_R_Overlap_2025-08-15_15:47/L1_R_Overlap_weights_E75.pt"
    L1_SMallRandom_Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_random_Overlap/models/NoStop_L1_R_SmallOverlap_2025-08-29_12:41/NoStop_L1_R_SmallOverlap_weights_E75.pt"
    L2_Random_Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT02_train_random_overlap/models/L2_R_Overlap_2025-08-15_15:47/L2_R_Overlap_weights_E75.pt"
    R4_Random_Overlap_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_RME04_train_random_overlap/models/R4_R_Overlap_2025-08-15_15:48/R4_R_Overlap_weights_E75.pt"
    L1_Normal_model= "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/No_Chipping_LMNT01_dataset/models/NoStop_No_Chipping_LMNT01_2025-08-29_12:43/NoStop_No_Chipping_LMNT01_weights_E75.pt"
    L2_Normal_model= "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/No_Chipping_LMNT02_dataset/models/NoStop_No_Chipping_LMNT02_2025-08-29_12:44/NoStop_No_Chipping_LMNT02_weights_E75.pt"
    R4_Normal_model= "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/No_Chipping_RME04_dataset/models/NoStop_No_Chipping_RME04_2025-08-29_12:43/NoStop_No_Chipping_RME04_weights_E75.pt"
    # Datasets
    L1_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Small_Eval_LMNT01"
    L2_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Small_Eval_LMNT02"
    R4_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Small_Eval_RME04"


    #Set the parameters of your evaluation here
    evaluation_params = {
        "experiment":"ImageChipping",
        "run_name": f"Evaluation_NoStop_L1-10_{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
        "model": L1_10Overlap_model,
        "dataset": L1_dataset,
        "device_no": 7,
        "architecture_type":"Panoptic_Sentinel" #Sentinel, SentinelNMS, Panoptic_Sentinel, 
        }

    metrics = [calculate_pr_curves, centroid_l2_accuracy]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(evaluation_params["experiment"])
    mlflow.end_run()
    with mlflow.start_run(run_name=evaluation_params["run_name"]):
        mlflow.log_params(evaluation_params)
        device = torch.device(f"cuda:{evaluation_params["device_no"]}" if torch.cuda.is_available() else "cpu")
        output_dir = os.path.join("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments", evaluation_params["run_name"])

        if evaluation_params["architecture_type"].lower() == "sentinel":
            model = Sentinel(normalize=False)
            model.load_original_model(evaluation_params["model"])
            model.to(device)
            model.eval()
        elif evaluation_params["architecture_type"].lower() == "sentinelnms":
            model = SentinelNMS(normalize=False)
            model.load_original_model(evaluation_params["model"])
            model.to(device)
            model.eval()
        elif evaluation_params["architecture_type"].lower() == "panoptic_sentinel":
            model = Sentinel_Panoptic(normalize=False)
            model.load_original_model(evaluation_params["model"])
            model.to(device)
            model.eval()
        else:
            model = Sentinel(normalize=False)
            model.load_original_model(evaluation_params["model"])
            model.to(device)
            model.eval()





        database = evaluate_on_test_sets(evaluation_params["dataset"], model, output_dir, evaluation_params["run_name"], metrics, device_no=device)
    mlflow.end_run()