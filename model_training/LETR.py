# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import mlflow
import os

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import os
import mlflow
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference, calculate_centroid_difference_with_confidence
from training_frameworks.evaluate_one_epoch import evaluate_LETR
from training_frameworks.train_one_epoch import train_LETR_one_epoch
from models.LETR.src.models import build_model
from models.LETR.src.args import get_args_parser


if __name__ == "__main__":


    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    #My own args   
    args.dataset_file='coco'
    args.coco_path='/home/davidchaparro/Datasets/ZScaledRME03AllStar'
    args.output_dir='/home/davidchaparro/Datasets/ZScaledRME03AllStar/checkpoint_models'
    args.resume=''
    args.start_epoch=0
    args.eval=False
    args.batch_size = 16
    args.epochs = 500

    train_params = {
        "epochs": 250,
        "batch_size": 42,
        "lr": 2e-4, #sqrt(batch_size)*4e-4
        "model_path": None,
        "training_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat_Training_Channel_Mixture_C/train",
        "validation_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat_Training_Channel_Mixture_C/val",
        "gpu": 3,
        "evaluation_metrics": [], 
        "momentum": 0.9,
        "weight_decay": 0.0005, 
        "experiment_name": "Star_Detection_LETR"
    }

    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Dataset paths
    training_dir = train_params["training_dir"]
    validation_dir = train_params["validation_dir"]
    base_dir = os.path.dirname(training_dir)
    models_dir = os.path.join(base_dir, "models", train_params["experiment_name"])
    os.makedirs(models_dir, exist_ok=True)

    # Load COCO-style dataset
    training_set = CocoDetection(root=training_dir, annFile=os.path.join(training_dir, "annotations", "annotations.json"), transforms=transform)
    validation_set = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
    training_loader = DataLoader(training_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
    validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))

    # Load model
    model, criterion, postprocessors = build_model(args)
    model_without_ddp = model
    if train_params["model_path"] is not None:
        model.load_state_dict(torch.load(train_params["model_path"] ))
        print(f"Loading Model: {train_params["model_path"]}")

    # Optimizer
    device = torch.device(f"cuda:{train_params["gpu"]}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Training Loop
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(train_params["experiment_name"])
    mlflow.end_run()
    print("Start training")
    start_time = time.time()
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_params(train_params)
        model.train()
        for epoch in range(train_params["epochs"]):
            losses = train_LETR_one_epoch(model, criterion, postprocessors, training_loader, optimizer, device, epoch, args.clip_max_norm, args)
            lr_scheduler.step()
            mlflow.log_metrics(losses, epoch) 
            results = evaluate_LETR(model, criterion, postprocessors, validation_loader, train_params["evaluation_metrics"], device)
            mlflow.log_metrics(results, epoch) 
            torch.save(model.state_dict(), os.path.join(models_dir,f"retinanet_weights_E{epoch}.pt"))
        mlflow.end_run()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
