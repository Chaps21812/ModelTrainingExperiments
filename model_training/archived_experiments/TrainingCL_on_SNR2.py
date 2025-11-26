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


if __name__ == "__main__":

    Real1Step = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1/train_TTS/train"]
    Real2Step = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_High_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_Low_TTS/train"]
    Real5Step = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_5_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_4_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_3_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_2_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_1_TTS/train",]
    SIM1Step = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim/train_TTS/train"]
    SIM2Step = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_High_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_Low_TTS/train"]
    SIM5STep = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_5_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_4_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_3_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_2_TTS/train",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_1_TTS/train",]
    Real1StepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1/train_TTS/val"]
    Real2StepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_High_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_Low_TTS/val"]
    Real5StepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_5_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_4_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_3_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_2_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_L1_1_TTS/val",]
    SIM1StepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim/train_TTS/val"]
    SIM2StepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_High_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_Low_TTS/val"]
    SIM5STepVal = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_5_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_4_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_3_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_2_TTS/val",
                "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CL_Sim_1_TTS/val",]

    training_sets = Real2Step
    validation_sets = Real2StepVal
    switch_every = 51
    epochs  = int(len(training_sets)*switch_every)

    train_params = {
        "project": "Curriculum_Learning_SNR",
        "experiment_name": f"CL-NoStopReal2Step{switch_every}",
        "epochs": epochs,
        "switch_every": switch_every,
        "batch_size": 42,
        "lr": 1e-4, #sqrt(batch_size)*4e-4
        "gpu": 7,
        "momentum": 0.9,
        "weight_decay": 0.0005, 
        "TConfidence": None,
        "TFit":None,
        "model_path": None,
        "training_dir": training_sets,
        "validation_dir": validation_sets,
        "evaluation_metrics": [centroid_l2_accuracy], 
        "early_stopping_metric":"bbox_regression", #bbox_regression, F1_tc-0.5_tf-1, None
        "patience_epochs":256,
    }

    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Dataset paths
    run_name = f"{train_params["experiment_name"]}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
    training_dir = train_params["training_dir"]
    validation_dir = train_params["validation_dir"]
    base_dir = os.path.dirname(training_dir[0])
    models_dir = os.path.join(base_dir, "models", run_name)
    train_params["model_save_dir"] = models_dir
    train_params["script_name"] = __file__
    os.makedirs(models_dir, exist_ok=True)

    # Load COCO-style dataset
    tsets = []
    tloaders = []
    for df in train_params["training_dir"]:
        training_set = CocoDetection(root=df, annFile=os.path.join(df, "annotations", "annotations.json"), transforms=transform)
        training_loader = DataLoader(training_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
        tsets.append(training_set)
        tloaders.append(training_loader)

    if isinstance(train_params["validation_dir"], str):
        validation_set = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
    else:
        vsets = []
        vloaders = []
        for dfv in train_params["validation_dir"]:
            validation_set = CocoDetection(root=dfv, annFile=os.path.join(dfv, "annotations", "annotations.json"), transforms=transform)
            validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
            vsets.append(validation_set)
            vloaders.append(validation_loader)

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
        epoch_counter = 0
        training_set = 0
        try:
            # Training Loop
            best_performance = 0
            best_loss = 1000
            waited_epochs = 0
            stopping_metric = train_params["early_stopping_metric"]
            model.train()
            for epoch in range(train_params["epochs"]):
                if epoch_counter >= train_params["switch_every"]:
                    best_performance = 0
                    best_loss = 1000
                    epoch_counter = 0
                    waited_epochs = 0
                    training_set += 1
                    mlflow.log_param(f"Dataset switch {training_set}", f"Dataset switch on epoch {epoch}")
                if training_set >= len(tloaders):
                    break
                training_loader = tloaders[training_set]
                if isinstance(train_params["validation_dir"], str):
                    validation_loader = validation_loader
                else:
                    validation_loader = vloaders[training_set]

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
                    best_performance = 0
                    best_loss = 1000
                    training_set += 1
                    epoch_counter = 0
                    waited_epochs = 0
                    mlflow.log_param(f"Dataset switch {training_set}", f"Dataset switch on epoch {epoch}")

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





