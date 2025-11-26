import torch
import torchvision
from torchvision.datasets import CocoDetection
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
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, ssd300_vgg16


if __name__ == "__main__":

    train_params = {
        "project": "SentinelDemo",
        "experiment_name": "RetinaNet-RME01-2025",
        "epochs": 256,
        "batch_size": 40,
        "lr": 1e-4, #sqrt(batch_size)*4e-4
        "gpu": 6,
        "momentum": 0.9,
        "weight_decay": 0.0005, 
        "TConfidence": None,
        "TFit":None,
        "model_path": None,
        "training_dir": "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME01_2025/train",
        "validation_dir": "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME01_2025/val",
        "testing_dir": "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME01_2025/test",
        "evaluation_metrics": [centroid_l2_accuracy], 
        "early_stopping_metric":"classification", #bbox_regression, F1_tc-0.5_tf-1, None
        "patience_epochs":30,
    }

    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Dataset paths
    run_name = f"{train_params["experiment_name"]}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
    training_dir = train_params["training_dir"]
    validation_dir = train_params["validation_dir"]
    testing_dir = train_params["testing_dir"]
    base_dir = os.path.dirname(training_dir)
    models_dir = os.path.join(base_dir, "models", run_name)
    train_params["model_save_dir"] = models_dir
    train_params["script_name"] = __file__
    os.makedirs(models_dir, exist_ok=True)

    # Load COCO-style dataset
    training_set = CocoDetection(root=training_dir, annFile=os.path.join(training_dir, "annotations", "annotations.json"), transforms=transform)
    validation_set = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
    testing_set = CocoDetection(root=testing_dir, annFile=os.path.join(testing_dir, "annotations", "annotations.json"), transforms=transform)
    training_loader = DataLoader(training_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
    validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
    testing_loader = DataLoader(testing_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))

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
            best_loss = torch.inf
            waited_epochs = 0
            stopping_metric = train_params["early_stopping_metric"]
            for epoch in range(train_params["epochs"]):
                test_results_dict = {}
                path = os.path.join(models_dir,f"{train_params["experiment_name"]}_weights_E{epoch}.pt")
                losses = train_one_epoch(model, optimizer, training_loader, device, epoch)
                check_loss_for_nans(losses, epoch)
                torch.save(model.state_dict(), path)
                results, validation_losses = retinaNet_evaluate(model, epoch, validation_loader, train_params, device)
                testing_results, testing_validation_losses = retinaNet_evaluate(model, epoch, testing_loader, train_params, device)
                for key, value in (testing_results | testing_validation_losses).items():
                    test_results_dict[f"Testset_{key}"]= value
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





