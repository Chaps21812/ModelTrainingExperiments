import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
import os
import mlflow
from tqdm import tqdm
from models.Sentinel_Models.Sentinel_Retina_Net_Stitch import Sentinel_Panoptic
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from evaluation.dataset_studies import Dataset_study
from training_frameworks.format_targets import format_targets_bboxes
import torch.nn.functional as F
from datetime import datetime


def evaluate_over_time(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
    total_targets= []
    total_predictions = []
    max_prediction_length = 0
    max_target_length = 0
    metrics = {}
    H=1
    W=1

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)
            H=images[0].shape[1]
            W=images[0].shape[2]

            targets = format_targets_bboxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            tensor_targets = model.output_formatter.convert_targets(targets)

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

    padded_predictions = [F.pad(t, pad=(0, max_target_length - t.shape[2])) for t in total_predictions]
    padded_targets = [F.pad(t, pad=(0, max_prediction_length - t.shape[2])) for t in total_targets]

    total_preds = torch.cat(padded_predictions, dim=0)
    total_tgts = torch.cat(padded_targets, dim=0)

    for metric in evaluation_metrics:
        results = metric(total_preds, total_tgts, image_height=(W,H))
        metrics.update(results)
    return metrics

def perform_timewise_benchmark(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5, database=None, batchsize=10):
    if database is None:
        database = Dataset_study(output_directory, datasets, experiment_title)
    else:
        database = database
    # Custom transforms (RetinaNet expects images and targets)
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.ToTensor(),
        # T.RandomHorizontalFlip(0.5),
        # T.Resize(1024),
    ])
    # Load model
    # model = torchvision.models.detection.retinanet_resnet50_fpn()
    # model.load_state_dict(torch.load(model_path))
    # print(f"Loading Model: {model_path}")
    model = Sentinel_Panoptic(normalize=False, sub_batch_size=batchsize)
    # model = Sentinel(normalize=False)
    model.load_original_model(model_path)
    model.to(device)
    model.eval()


    mlflow.log_param("Model", model_path)
    mlflow.log_param("Dataset", datasets)

    for index, path in tqdm(enumerate(database)):
        # Load COCO-style dataset
        validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
        validation_loader = DataLoader(validation_set, batch_size=16, shuffle=True, collate_fn=lambda x: (zip(*x)))
        results = evaluate_over_time(model, path, index,validation_loader, evaluation_metrics, device)
        mlflow.log_metrics(results, index)
        results["date"] = database.path_to_date[path]
        database.add_metric(results)
        database.save()

        # Optimizer
    database.plot_all_metrics()
    return database

def perform_cumulative_evaluation(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5, database=None, batchsize=10):
    if database is None:
        database = Dataset_study(output_directory, datasets, experiment_title)
    else:
        database = database
    # Custom transforms (RetinaNet expects images and targets)
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.ToTensor(),
        # T.RandomHorizontalFlip(0.5),
        # T.Resize(1024),
    ])
    # Load model
    # model = torchvision.models.detection.retinanet_resnet50_fpn()
    # model.load_state_dict(torch.load(model_path))
    # print(f"Loading Model: {model_path}")
    model = Sentinel_Panoptic(normalize=False, sub_batch_size=batchsize)
    model.load_original_model(model_path)
    model.to(device)
    model.eval()

    # Load COCO-style dataset
    validation_sets = []
    for validation_dir in datasets:
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_sets.append(temp_validation)

    validation_set = ConcatDataset(validation_sets)
    validation_loader = DataLoader(validation_set, batch_size=16, shuffle=True, collate_fn=lambda x: (zip(*x)))

    mlflow.log_param("Model", model_path)
    mlflow.log_param("Dataset", datasets)

    results = evaluate_over_time(model, "", 0,validation_loader, evaluation_metrics, device)
    database.plot_all_metrics()
    mlflow.log_params(results)
    database.cumulative_metrics = results
    database.save_cumulative_metrics()
    database.save()
    return database

if __name__ == "__main__":
    LMNT01_Evaluation_set = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-13_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-25_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-06_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-15_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-23_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-15_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-26_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-06_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-17_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-20_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-30_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-07_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-10_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-23_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-03_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-10_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-16_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-25_Channel_Mixture_C_Eval"
    ]
    LNT02_Evaluation_set = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-06_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-14_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-08_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-10_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-31_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-14_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-19_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-07_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-24_Channel_Mixture_C_Eval", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-31_Channel_Mixture_C_Eval"
    ]
    RME04_Evaluation_set = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-04_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-05_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-06_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-07_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-08_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-09_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-11_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-12_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-13_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-15_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-28_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-07-09_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-07-19_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2025-06-04_Channel_Mixture_C"]
    RME04_Evaluation_set_No_Future = ["/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-04_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-05_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-06_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-07_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-08_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-09_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-11_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-12_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-13_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-15_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-06-28_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-07-09_Channel_Mixture_C", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04Sat-2024-07-19_Channel_Mixture_C"]


    #Pull data from these variables
    # Panoptic_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/PanopticL1E49.pt"
    datasets = [RME04_Evaluation_set_No_Future,RME04_Evaluation_set, LNT02_Evaluation_set, LMNT01_Evaluation_set]

    #Adjust this code only
    run_name = f"PanopticL1Sim_on_L2_Data"
    Panoptic_model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Panoptic_MC_L1Sim_train/models/L1Sim/retinanet_weights_E45.pt"
    dataset = LNT02_Evaluation_set
    device_no = 6
    batchsize = 120
    #Adjust this code only



    model = Panoptic_model
    metrics = [centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference]
    try:
        # Training Loop
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Panoptic Sentinel Over Time")
        run_name = f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
        with mlflow.start_run(run_name=run_name):
            output_dir = os.path.join("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments", run_name)
            database = perform_timewise_benchmark(dataset, model, output_dir, run_name, metrics, device_no=device_no, batchsize=batchsize)
            database = perform_cumulative_evaluation(dataset, model, output_dir, run_name, metrics, device_no=device_no, database=database, batchsize=batchsize)
        mlflow.end_run()
    except Exception as e:
        # Log the exception message as a tag or param
        mlflow.log_param("run_status", "FAILED")
        mlflow.log_param("error_type", type(e).__name__)
        mlflow.log_param("error_message", str(e))
        # Optionally re-raise if you want the program to crash
        raise


