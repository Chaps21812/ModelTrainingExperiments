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
from preprocessing_techniques.preprocessing import iqr_log_16bit, raw_file_16bit
import re


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
            "experiment_name": f"Raw16-AllData-NoResize",
            "epochs": 250,
            "batch_size": 42,
            "lr": 1e-4, #sqrt(batch_size)*4e-4
            "gpu": 2,
            "momentum": 0.9,
            "weight_decay": 0.0005, 
            "TConfidence": None,
            "TFit":None,
            "model_path": None,
            "min_size": 512,
            "max_size": 10000,
            "main_directories": ALL_TELESCOPES,
            "data_limit":200000,
            "preprocess_func":raw_file_16bit.__name__,
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

        merged_trains = [x for group in zip_longest(*sub_directories_train, fillvalue=None) for x in group if x is not None]
        merged_vals = [x for group in zip_longest(*sub_directories_val, fillvalue=None) for x in group if x is not None]

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
        model =  torchvision.models.detection.retinanet_resnet50_fpn( min_size=train_params["min_size"],max_size=train_params["max_size"])
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

                mlflow.end_run()
            except Exception as e:
                # Log the exception message as a tag or param
                mlflow.log_param("run_status", "FAILED")
                mlflow.log_param("error_type", type(e).__name__)
                mlflow.log_param("error_message", str(e))
                raise





