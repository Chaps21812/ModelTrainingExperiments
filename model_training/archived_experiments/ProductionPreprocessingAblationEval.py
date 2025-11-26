import torch
import torch
import torchvision
import itertools
import json
# from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import os
import mlflow
from datasets.coco_tools import COCODataset
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
from preprocessing_techniques.preprocessing import iqr_log_16bit
from preprocessing_techniques.coco_data_loader import Coco16bitGray, collate_fn, CocoDetection
from evaluation.dataset_studies import Single_Dataset_Study
from preprocessing_techniques.preprocessing import raw_file, raw_file_16bit, iqr_log, iqr_log_16bit, iqr_clipped, iqr_clipped_16bit, zscale, zscale_16bit

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
        for images, targets, image_id in tqdm(dataloader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)

            original_tgt_attributes.extend(targets)
            targets = format_targets_bboxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            tensor_targets = model.output_formatter.convert_targets(targets)

            image_ids.extend(image_id)
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

def evaluate_on_test_sets(datasets:Union[str,list], model, output_directory:str, experiment_title, evaluation_metrics, device_no:torch.device, evaluation_params=None, database=None):
    if isinstance(datasets, str):
        datasets = [datasets]
    if database is None:
        database = Single_Dataset_Study(output_directory, datasets, experiment_title)
    else:
        database = database

    multiframe_dictionary = {}
    image_id_to_collect = {}
    image_id_to_position = {}
    for dataset in datasets:
        with open(os.path.join(dataset, "multiframe_annotations", "multiframe_annotations.json"),'r') as f:
            multiframe_data = json.load(f)
            for index, (collect_id,info) in enumerate(multiframe_data.items()):
                multiframe_dictionary[collect_id] = [{} for i in range(len(info))]
                for frame in info:
                    image_id_to_collect[frame["image_id"]] = collect_id
                    image_id_to_position[frame["image_id"]] = frame["order"]

    transform = T.Compose([
        T.ToTensor(),
    ])

    if "16" in evaluation_params["preprocessing"].__name__:
        validation_coco = Coco16bitGray(datasets[0], transform=transform)
    else:
        val__path = datasets[0]
        val_ann_path = os.path.join(val__path, "annotations", "annotations.json")
        validation_coco = CocoDetection(val__path, annFile=val_ann_path, transform=transform)

    
    validation_loader = DataLoader(validation_coco, batch_size=40, shuffle=False, collate_fn=lambda x: (zip(*x)))

    results, predictions, gts, image_ids = evaluate_single_dataset(model,validation_loader, device)
    database.add_metrics(results)
    database.add_gts(gts)
    database.add_predictions(predictions)
    database.add_image_ids(image_ids)
    database.save()

    for pred,id in zip(predictions, image_ids):
        trans  = pred.transpose(0,1)
        output_prediction = []
        for col in trans:
            if not torch.all(col == 0):
                single_prediction = {
                    "minimumXPixel": float(col[0]-col[2]/2),
                    "maximumXPixel": float(col[0]+col[2]/2),
                    "minimumYPixel": float(col[1]-col[3]/2),
                    "maximumYPixel": float(col[1]+col[3]/2),
                    "confidenceValueFloat": float(col[4])
                }
                output_prediction.append(single_prediction)
        collect_id = image_id_to_collect[int(id)]
        order = image_id_to_position[int(id)]
        multiframe_dictionary[collect_id][order] = output_prediction

    for dataset in datasets:
        os.makedirs(os.path.join(dataset, "model_outputs", evaluation_params["run_name"]), exist_ok=True)
        with open(os.path.join(dataset, "model_outputs", evaluation_params["run_name"], "model_outputs.json"),'w') as f:
            json.dump(multiframe_dictionary, f, indent=4)

    only_params = {k: v for k, v in results.items() if isinstance(v, float)}
    mlflow.log_params(only_params)
    return database

if __name__ == "__main__":
    

    
    # preprocess_funcs = [raw_file_16bit, raw_file, iqr_log, iqr_log_16bit, iqr_clipped, iqr_clipped_16bit, zscale, zscale_16bit]
    preprocess_funcs = [raw_file, raw_file_16bit, iqr_log, iqr_log_16bit]

    MODEL_A = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/models/RME01-raw_file_2025-11-10_05:03/RME01-raw_file_weights_E56.pt"
    MODEL_B = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/models/RME01-raw_file_16bit_2025-11-07_14:53/RME01-raw_file_16bit_weights_E109.pt"
    MODEL_C = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/models/RME01-iqr_log_2025-11-12_00:17/RME01-iqr_log_weights_E55.pt"
    MODEL_D = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/models/RME01-iqr_log_16bit_2025-11-17_11:06/RME01-iqr_log_16bit_weights_E58.pt"

    models = [MODEL_A,MODEL_B,MODEL_C,MODEL_D]
    for model, process_funx in zip(models,preprocess_funcs):
        #Set the parameters of your evaluation here
        evaluation_params = {
            "experiment":"Sentinel_Preprocessing_Experiments_2",
            "run_name": f"Eval_preprocess_{process_funx.__name__}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
            "model": model,
            "dataset": "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025Data/test",
            "device_no": 7,
            "preprocessing": process_funx,
            "architecture_type":"Sentinel", #Sentinel, SentinelNMS, Panoptic_Sentinel, ,
            "pid": os.getpid(),
            "min_size": 800,
            "max_size": 1333,
            "recentroid":True,
            }
        
        print(os.getpid())
        COCODataset(os.path.dirname(evaluation_params["dataset"])).convert_test_images(process_funx)

        metrics = [calculate_pr_curves, centroid_l2_accuracy]
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(evaluation_params["experiment"])
        mlflow.end_run()
        with mlflow.start_run(run_name=evaluation_params["run_name"]):
            mlflow.log_params(evaluation_params)
            device = torch.device(f"cuda:{evaluation_params["device_no"]}" if torch.cuda.is_available() else "cpu")
            output_dir = os.path.join("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments", evaluation_params["run_name"])

            if evaluation_params["architecture_type"].lower() == "sentinel":
                model = Sentinel(normalize=False, min_size=evaluation_params["min_size"], max_size=evaluation_params["max_size"])
                model.load_original_model(evaluation_params["model"])
                model.to(device)
                model.eval()
            elif evaluation_params["architecture_type"].lower() == "sentinelnms":
                model = SentinelNMS(normalize=False, min_size=evaluation_params["min_size"], max_size=evaluation_params["max_size"])
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





            database = evaluate_on_test_sets(evaluation_params["dataset"], model, output_dir, evaluation_params["run_name"], metrics, device_no=device, evaluation_params=evaluation_params)
            study = Single_Dataset_Study.load(output_dir)
            study.plot_PR_Curves()
            study.plot_per_attribute_PR("local_snr", 20, log_y=True)
            study.plot_per_attribute_PR("x_center", 20)
            study.plot_per_attribute_PR("y_center", 20)
        mlflow.end_run()