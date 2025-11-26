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
from evaluation.evaluation_metrics import centroid_l2_accuracy
from training_frameworks.evaluate_one_epoch import retinaNet_evaluate
from training_frameworks.train_one_epoch import train_one_epoch
from training_frameworks.nan_detection import check_loss_for_nans
from preprocessing_techniques.coco_data_loader import Coco16bitGray, collate_fn


if __name__ == "__main__":

    data_limit = [1000,5000,10000,15000,20000,30000]

    for DL in data_limit:
        train_params = {
            "project": "Production_Experiments",
            "experiment_name": f"RME01-{DL}",
            "epochs": 250,
            "batch_size": 36,
            "lr": 1e-4, #sqrt(batch_size)*4e-4
            "gpu": 4,
            "momentum": 0.9,
            "weight_decay": 0.0005, 
            "TConfidence": None,
            "TFit":None,
            "model_path": None,
            "main_directory": "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations",
            "data_limit":None,
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
        base_dir = os.path.dirname(train_params["main_directory"])
        models_dir = os.path.join(base_dir, "models", run_name)
        train_params["model_save_dir"] = models_dir
        train_params["script_name"] = __file__
        os.makedirs(models_dir, exist_ok=True)

        # Load COCO-style dataset
        train_paths = []
        val_paths = []
        training_dates = []
        train_count = 0
        val_count = 0
        training_cocos = []
        validation_cocos = []

        folders = os.listdir(train_params["main_directory"])
        folders_sorted = sorted(
            folders,
            key=lambda f: datetime.strptime(f, "%Y-%m-%d") 
        )

        for folder in folders_sorted:
            if "2024" in folder:
                continue
            training_dates.append(folder)
            if os.path.isdir(os.path.join(train_params["main_directory"], folder)):
                train_paths.append(os.path.join(train_params["main_directory"], folder, "train"))
                val_paths.append(os.path.join(train_params["main_directory"], folder, "val"))
                train_images = len(os.listdir(os.path.join(train_params["main_directory"], folder, "train", "images")))
                val_images = len(os.listdir(os.path.join(train_params["main_directory"], folder, "val", "images")))
                train_count += train_images
                val_count += val_images
                training_cocos.append(Coco16bitGray(os.path.join(train_params["main_directory"], folder, "train"), transform=transform))
                validation_cocos.append(Coco16bitGray(os.path.join(train_params["main_directory"], folder, "val"), transform=transform))
                if train_params["data_limit"] is not None and train_count > train_params["data_limit"]:
                    break
        train_params["measured_train_count"] = train_count
        train_params["measured_val_count"] = val_count
        train_params["training_dates"] = training_dates

        # Merge them
        merged_training = ConcatDataset(training_cocos)
        merged_validation = ConcatDataset(validation_cocos)

        training_loader = DataLoader(merged_training, batch_size=30, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(merged_validation, batch_size=30, shuffle=True, collate_fn=collate_fn)

        # Load model
        model =  torchvision.models.detection.retinanet_resnet50_fpn()
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





