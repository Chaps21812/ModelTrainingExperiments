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
    # CL1L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1/train_TTS/models/CL-Real1Step51_2025-08-20_08:46/CL-Real1Step51_weights_E50.pt"
    # CL2L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_High_TTS/models/CL-Real2Step51_2025-08-20_08:46/CL-Real2Step51_weights_E82.pt"
    # CL5L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_5_TTS/models/CL-Real5Step51_2025-08-21_03:44/CL-Real5Step51_weights_E164.pt"
    # CL1Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim/train_TTS/models/CL-Sim1Step51_2025-08-20_08:46/CL-Sim1Step51_weights_E50.pt"
    # CL2Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_High_TTS/models/CL-Sim2Step51_2025-08-22_05:04/CL-Sim2Step51_weights_E101.pt"
    # CL5Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_5_TTS/models/CL-Sim5Step51_2025-08-25_10:54/CL-Sim5Step51_weights_E227.pt"
    CL1L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1/train_TTS/models/CL-NoStopReal1Step51_2025-08-26_10:09/CL-NoStopReal1Step51_weights_E50.pt"
    CL2L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_High_TTS/models/CL-NoStopReal2Step51_2025-08-26_10:09/CL-NoStopReal2Step51_weights_E101.pt"
    CL5L1_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_5_TTS/models/CL-NoStopReal5Step51_2025-08-26_06:17/CL-NoStopReal5Step51_weights_E254.pt"
    CL1Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim/train_TTS/models/CL-NoStopSim1Step51_2025-08-26_06:17/CL-NoStopSim1Step51_weights_E50.pt"
    CL2Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_High_TTS/models/CL-NoStopSim2Step51_2025-08-26_06:17/CL-NoStopSim2Step51_weights_E101.pt"
    CL5Sim_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_5_TTS/models/CL-NoStopSim5Step51_2025-08-26_06:17/CL-NoStopSim5Step51_weights_E254.pt"

    Sim_testing_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim/test"
    Real_testing_dataset = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1/test"
    #Set the parameters of your evaluation here
    evaluation_params = {
        "experiment":"Curriculum_Learning_SNR",
        "run_name": f"Evaluation_Sim2Real_{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
        "model": CL2Sim_model,
        "dataset": Sim_testing_dataset,
        "device_no": 5,
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