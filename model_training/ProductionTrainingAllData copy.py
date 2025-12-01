import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import ConcatDataset
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from datetime import datetime
import os
import mlflow
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from itertools import zip_longest
from evaluation.evaluation_metrics import centroid_l2_accuracy
from training_frameworks.evaluate_one_epoch import retinaNet_evaluate
from training_frameworks.train_one_epoch import train_one_epoch
from training_frameworks.nan_detection import check_loss_for_nans
from preprocessing_techniques.coco_data_loader import Coco16bitGray, collate_fn
from preprocessing_techniques.preprocessing import iqr_log_16bit
import re


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

def evaluate_on_test_sets(datasets:Union[str,list], model, output_directory:str, experiment_title, evaluation_metrics, device_no:torch.device, evaluation_params=None, database=None, test_data_loader=None):
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


    if test_data_loader is not None:
        validation_loader = test_data_loader
    else:
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


def auto_eval_all_tests(model:str,test_loader:DataLoader, epoch:int, params:dict):
    D1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-RME04-2025"
    D2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-ABQ01-2025"
    D3 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsats-LMNT02-2024"
    D4 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsats-LMNT01-2024"
    
    datasets = [D1,D2,D3,D4]
    for dataset in datasets:
        metrics = [calculate_pr_curves, centroid_l2_accuracy]
        device = torch.device(f"cuda:{params["gpu"]}" if torch.cuda.is_available() else "cpu")
        output_dir = os.path.join("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments", params["experiment_name"])

        if params["architecture_type"].lower() == "sentinel":
            model = Sentinel(normalize=False, min_size=params["min_size"], max_size=params["max_size"])
            model.load_original_model(params["model"])
            model.to(device)
            model.eval()
        elif params["architecture_type"].lower() == "sentinelnms":
            model = SentinelNMS(normalize=False, min_size=params["min_size"], max_size=params["max_size"])
            model.load_original_model(params["model"])
            model.to(device)
            model.eval()
        elif params["architecture_type"].lower() == "panoptic_sentinel":
            model = Sentinel_Panoptic(normalize=False)
            model.load_original_model(params["model"])
            model.to(device)
            model.eval()
        else:
            model = Sentinel(normalize=False)
            model.load_original_model(params["model"])
            model.to(device)
            model.eval()

        database = evaluate_on_test_sets(params["dataset"], model, output_dir, params["run_name"], metrics, device_no=device, evaluation_params=params, test_data_loader=test_loader)
        study = Single_Dataset_Study.load(output_dir)
        study.plot_PR_Curves()
        study.plot_per_attribute_PR("local_snr", 20, log_y=True)
        study.plot_per_attribute_PR("x_center", 20)
        study.plot_per_attribute_PR("y_center", 20)


if __name__ == "__main__":
    TELESCOPE_A = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025-Annotations"
    TELESCOPE_B = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations"
    TELESCOPE_C = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT02-2025-Annotations"
    TELESCOPE_D = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01-2025-Annotations"
    TELESCOPE_E = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/ABQ01-2025-Annotations"
    
    ALL_TELESCOPES = [TELESCOPE_A,TELESCOPE_B, TELESCOPE_C, TELESCOPE_D,TELESCOPE_E]

    data_limit = [100000]

    for DL in data_limit:
        train_params = {
            "project": "Production_Experiments",
            "experiment_name": f"Production-AllData-Resizing-raw16",
            "epochs": 250,
            "batch_size": 42,
            "lr": 1e-4, #sqrt(batch_size)*4e-4
            "gpu": 2,
            "momentum": 0.9,
            "weight_decay": 0.0005, 
            "TConfidence": None,
            "TFit":None,
            "model_path": None,
            "main_directories": ALL_TELESCOPES,
            "data_limit":200000,
            "min_size": 800,
            "max_size": 1333,
            "preprocess_func":iqr_log_16bit.__name__,
            "evaluation_metrics": [centroid_l2_accuracy], 
            "early_stopping_metric":"classification", #bbox_regression, F1_tc-0.5_tf-1, None
            "patience_epochs":30,
            "PID":os.getpid()
        }


        # Custom transforms (RetinaNet expects images and targets)
        transform = T.Compose([
            T.ToTensor(),
        ])

        # Dataset paths
        run_name = f"{train_params["experiment_name"]}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
        base_dir = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets"
        models_dir = os.path.join(base_dir, "models", run_name)
        train_params["model_save_dir"] = models_dir
        train_params["script_name"] = __file__
        os.makedirs(models_dir, exist_ok=True)

        # Load COCO-style dataset


        train_count = 0
        val_count = 0
        training_cocos = []
        validation_cocos = []
        training_directories = []

        sub_directories_train = [[] for i in range(len(train_params["main_directories"]))]
        sub_directories_val = [[] for i in range(len(train_params["main_directories"]))]
        sub_directories_test = [[] for i in range(len(train_params["main_directories"]))]

        for j,path in enumerate(sub_directories_test):
            test_count += len(os.listdir(os.path.join(path, "images")))
            if "16" in train_params["preprocess_func"]:
                test_coco = Coco16bitGray(path, transform=transform)
            else:
                test_path = path
                test_ann_path = os.path.join(test_path, "annotations", "annotations.json")
                test_coco = CocoDetection(val__path, annFile=test_ann_path, transform=transform)

            training_cocos.append(test_coco)
            train_params["measured_test_count"] = test_count

        merged_test = ConcatDataset(training_cocos)
        test_loader = DataLoader(merged_training, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn)


        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for i, data_dir in enumerate(train_params["main_directories"]):
            folders = os.listdir(data_dir)
            valid_dates = [date for date in folders if pattern.match(date)]
            folders_sorted = sorted(
                valid_dates,
                key=lambda f: datetime.strptime(f, "%Y-%m-%d"))
            for folder in folders_sorted:
                if "2024" in folder:
                    continue
                if os.path.isdir(os.path.join(data_dir, folder)):
                    sub_directories_train[i].append(os.path.join(data_dir, folder, "train"))
                    sub_directories_val[i].append(os.path.join(data_dir, folder, "val"))
                    sub_directories_test[i].append(os.path.join(data_dir, folder, "test"))

        merged_trains = [x for group in zip_longest(*sub_directories_train, fillvalue=None) for x in group if x is not None]
        merged_vals = [x for group in zip_longest(*sub_directories_val, fillvalue=None) for x in group if x is not None]
        merged_test = [x for group in zip_longest(*sub_directories_test, fillvalue=None) for x in group if x is not None]

        for j,path in enumerate(merged_trains):
            training_directories.append(path)
            train_count += len(os.listdir(os.path.join(path, "images")))
            val_count += len(os.listdir(os.path.join(merged_vals[j], "images")))

            if "16" in train_params["preprocess_func"]:
                training_coco = Coco16bitGray(path, transform=transform)
                validation_coco = Coco16bitGray(merged_vals[j], transform=transform)
            else:
                train__path = path
                val__path = merged_vals[j]
                train_ann_path = os.path.join(train__path, "annotations", "annotations.json")
                val_ann_path = os.path.join(val__path, "annotations", "annotations.json")

                training_coco = CocoDetection(train__path, annFile=train_ann_path, transform=transform)
                validation_coco = CocoDetection(val__path, annFile=val_ann_path, transform=transform)

            training_cocos.append(training_coco)
            validation_cocos.append(validation_coco)
            if train_params["data_limit"] is not None and train_count > train_params["data_limit"]:
                break


        train_params["measured_train_count"] = train_count
        train_params["measured_val_count"] = val_count
        train_params["training_dates"] = training_directories

        # Merge them
        merged_training = ConcatDataset(training_cocos)
        merged_validation = ConcatDataset(validation_cocos)

        training_loader = DataLoader(merged_training, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(merged_validation, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn)

        # Load model
        model =  torchvision.models.detection.retinanet_resnet50_fpn(min_size=train_params["min_size"], max_size=train_params["max_size"])
        if train_params["model_path"] is not None:
            model.load_state_dict(torch.load(train_params["model_path"] ))
            print(f"Loading Model: {train_params["model_path"]}")

        # Optimizer
        device = torch.device(f"cuda:{train_params["gpu"]}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=train_params['lr'], momentum=train_params["momentum"], weight_decay=train_params["weight_decay"])

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(train_params["project"])
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(train_params)
            try:
                # Training Loop
                model.train()
                best_performance = 0
                best_loss = 1000
                waited_epochs = 0
                stopping_metric = train_params["early_stopping_metric"]
                for epoch in range(train_params["epochs"]):
                    path = os.path.join(models_dir,f"{train_params["experiment_name"]}_weights_E{epoch}.pt")
                    losses = train_one_epoch(model, optimizer, training_loader, device, epoch)
                    check_loss_for_nans(losses, epoch)
                    torch.save(model.state_dict(), path)
                    results, validation_losses = retinaNet_evaluate(model, epoch, validation_loader, train_params, device)
                    auto_eval_all_tests(model, test_loader, epoch, train_params)
                    if stopping_metric is not None:
                        waited_epochs += 1
                    if stopping_metric is not None and stopping_metric in results:
                        if results[stopping_metric] > best_performance:
                            waited_epochs = 0
                            best_performance = results[stopping_metric]
                    elif stopping_metric is not None and stopping_metric in validation_losses:
                        if validation_losses[stopping_metric] < best_loss:
                            waited_epochs = 0
                            best_loss = validation_losses[stopping_metric].item()
                    training_losses = {f"training_{k}":v for k,v in losses.items()}
                    processed_losses = {f"validation_{k}":v for k,v in validation_losses.items()}
                    mlflow.log_metrics(results, epoch) 
                    mlflow.log_metrics(training_losses, epoch) 
                    mlflow.log_metrics(processed_losses, epoch) 
                    if waited_epochs >= train_params["patience_epochs"]:
                        mlflow.log_param("run_status", f"Completed: Early Stop on Epoch {epoch}")
                        break


                    # mlflow.pytorch.log_model(model, artifact_path=path)
                    # # Register it in the Model Registry
                    # result = mlflow.register_model(
                    #     model_uri=f"runs:/{mlflow.active_run().info.run_id}/{path}",
                    #     name=f"retinanet_weights_E{epoch}.pt")
                mlflow.end_run()
            except Exception as e:
                # Log the exception message as a tag or param
                mlflow.log_param("run_status", "FAILED")
                mlflow.log_param("error_type", type(e).__name__)
                mlflow.log_param("error_message", str(e))
                raise





