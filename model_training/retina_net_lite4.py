import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
import os
import mlflow
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference, calculate_centroid_difference_10_confidence, calculate_centroid_difference_90_confidence
from training_frameworks.evaluate_one_epoch import evaluate
from training_frameworks.train_one_epoch import train_one_epoch


if __name__ == "__main__":

    train_params = {
        "epochs": 250,
        "batch_size": 24,
        "lr": 5e-5, #sqrt(batch_size)*4e-4
        "model_path": None,
        "training_dir": ["/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat_Training_Channel_Mixture_C/train","/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/train","/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat_Training_Channel_Mixture_C/train"
],
        "validation_dir": ["/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat_Training_Channel_Mixture_C/val","/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat_Training_Channel_Mixture_C/val","/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat_Training_Channel_Mixture_C/val"],
        "gpu": 6,
        "evaluation_metrics": [centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference, calculate_centroid_difference_10_confidence, calculate_centroid_difference_90_confidence], 
        "momentum": 0.9,
        "weight_decay": 0.0005, 
        "experiment_name": "All_Telescope_training"
    }

    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Dataset paths
    training_dir = train_params["training_dir"][0]
    validation_dir = train_params["validation_dir"][0]
    base_dir = os.path.dirname(training_dir)
    models_dir = os.path.join(base_dir, "models", train_params["experiment_name"])
    os.makedirs(models_dir, exist_ok=True)

    # Load COCO-style dataset
    training_sets = []
    validation_sets = []
    for training_set,validation_set in zip(train_params["training_dir"], train_params["validation_dir"]):
        temp_training = CocoDetection(root=training_dir, annFile=os.path.join(training_dir, "annotations", "annotations.json"), transforms=transform)
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        training_sets.append(temp_training)
        validation_sets.append(temp_validation)

    training_set = ConcatDataset(training_sets)
    validation_set = ConcatDataset(validation_sets)

    training_loader = DataLoader(training_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
    validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))

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

    # Training Loop
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(train_params["experiment_name"])
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_params(train_params)
        model.train()
        for epoch in range(train_params["epochs"]):
            losses = train_one_epoch(model, optimizer, training_loader, device, epoch)
            mlflow.log_metrics(losses, epoch) 
            results = evaluate(model, validation_dir, epoch, validation_loader, train_params["evaluation_metrics"], device)
            mlflow.log_metrics(results, epoch) 
            torch.save(model.state_dict(), os.path.join(models_dir,f"retinanet_weights_E{epoch}.pt"))
        mlflow.end_run()



