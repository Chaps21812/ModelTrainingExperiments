import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from datetime import datetime
import os
import mlflow
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset, DataLoader
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from sklearn.model_selection import GroupKFold
import itertools
import numpy as np
from evaluation.evaluation_metrics import centroid_l2_accuracy
from training_frameworks.evaluate_one_epoch import retinaNet_evaluate
from training_frameworks.train_one_epoch import train_one_epoch
from training_frameworks.nan_detection import check_loss_for_nans_bool
from training_frameworks.nan_detection import check_loss_for_nans


if __name__ == "__main__":


    training_sets = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_High_SNR_TTS/train","/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_SatSim_Low_SNR_TTS/train", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_High_SNR_TTS/train","/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Low_SNR_TTS/train"]
    validation_sets = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_LMNT01_train_random_Overlap/val"

    switch_every = 5
    epochs  = int(len(training_sets)*switch_every)

    train_params = {
        "project": "TEsting",
        "experiment_name": f"SRHL-BS{switch_every}",
        "k-folds":5,
        "epochs": epochs,
        "switch_every": switch_every,
        "batch_size": 36,
        "gpu": 6,
        "learning_rates":[1e-5, 5e-5, 1e-4, 3e-4],
        "weight_decays":[0.0001,0.0005,0.001],
        "momentums":[0.2,0.9,1.5],
        "TConfidence": None,
        "TFit":None,
        "model_path": None,
        "training_dir": training_sets,
        "validation_dir": validation_sets,
        "evaluation_metrics": [centroid_l2_accuracy], 
        "early_stopping_metric":"bbox_regression", #bbox_regression, F1_tc-0.5_tf-1, None
        "patience_epochs":10,
    }


    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Dataset paths
    run_name = f"KFold_{train_params["experiment_name"]}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
    training_dir = train_params["training_dir"]
    base_dir = os.path.dirname(training_dir)
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

    super_groups = []
    super_gkf = []
    for tset in tsets:
        groups = []
        for idx in range(len(tset)):
            img_id = tset.ids[idx] 
            img_info = tset.coco.imgs[img_id]
            collect_id = img_info.get("collect_id")
            groups.append(collect_id)
        gkf = GroupKFold(n_splits=train_params["k-folds"])
        super_groups.append(groups)
        super_gkf.append(gkf)


    # Optimizer
    device = torch.device(f"cuda:{train_params["gpu"]}" if torch.cuda.is_available() else "cpu")

    kfold_results = []

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(train_params["project"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(train_params)
        try:
            for lr, wd, mom in itertools.product(train_params["learning_rates"], train_params["weight_decays"], train_params["momentums"]):
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(gkf.split(range(len(training_set)), groups=groups)):
                    # Subsets
                    train_subset = Subset(training_set, [int(i) for i in train_idx])
                    val_subset   = Subset(training_set, [int(i) for i in val_idx])

                    # DataLoaders
                    collate_fn = lambda batch: tuple(zip(*batch))
                    training_loader = DataLoader(train_subset, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn)
                    val_loader   = DataLoader(val_subset, batch_size=train_params["batch_size"], shuffle=False, collate_fn=collate_fn)

                    # Load model
                    model =  torchvision.models.detection.retinanet_resnet50_fpn()
                    if train_params["model_path"] is not None:
                        model.load_state_dict(torch.load(train_params["model_path"] ))
                        print(f"Loading Model: {train_params["model_path"]}")
                    model.to(device)
                    params = [p for p in model.parameters() if p.requires_grad]
                    optimizer = torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
                    print(f"\nTesting params: lr={lr}, wd={wd}, momentum={mom}")

                    # Training Loop
                    model.train()

                    # Training Loop
                    epoch_counter = 0
                    training_set = 0
                    best_performance = 0
                    best_loss = 1000
                    waited_epochs = 0
                    stopping_metric = train_params["early_stopping_metric"]
                    model.train()
                    for epoch in range(train_params["epochs"]):
                        if epoch_counter >= train_params["switch_every"]:
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
                            training_set += 1
                            epoch_counter = 0
                            waited_epochs = 0
                            mlflow.log_param(f"Dataset switch {training_set}", f"Dataset switch on epoch {epoch}")


                    for epoch in range(train_params["epochs"]):
                        path = os.path.join(models_dir,f"{train_params["experiment_name"]}_weights_E{epoch}.pt")
                        losses = train_one_epoch(model, optimizer, training_loader, device, epoch)
                        has_nan = check_loss_for_nans_bool(losses, epoch)
                        if has_nan: 
                            fold_scores.append(0)
                            break
                        mlflow.log_metrics(losses, epoch) 
                    results = retinaNet_evaluate(model, epoch, val_loader, train_params, device)
                    mlflow.log_metrics(results, epoch)




                    score = results["F1_tc-0.1_tf-1"]
                    fold_scores.append(score)
                    print(f"  Fold {fold+1}: {score:.4f}")
                mean_score = np.mean(fold_scores)
                kfold_results.append(((lr, wd, mom), mean_score))
                print(f"Mean CV score: {mean_score:.4f}")
            best_params, best_score = max(results, key=lambda x: x[1])
            print("\nBest params:", best_params)
            print("Best CV score:", best_score)
            mlflow.log_param("Best Parameters Order","(lr, wd, momentum)" )
            mlflow.log_param("Best Parameters",best_params )
            mlflow.log_param("Best score",best_score )

            mlflow.end_run()
        except Exception as e:
            # Log the exception message as a tag or param
            mlflow.log_param("run_status", "FAILED")
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.log_param("error_message", str(e))
            raise

