import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import os
import mlflow
from tqdm import tqdm
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from evaluation.dataset_studies import Dataset_study
from training_frameworks.format_targets import format_targets_bboxes


def evaluate_over_time(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
    total_targets= []
    total_predictions = []
    metrics = {}

    model.eval()
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

        if epoch%10 == 0 and plot:
            plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
            # plot=False

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics

def perform_cumulative_evaluation(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5):
    database = Dataset_study(output_directory, datasets, experiment_title)
    # Custom transforms (RetinaNet expects images and targets)
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.ToTensor(),
        # T.RandomHorizontalFlip(0.5),
        # T.Resize(1024),
    ])
    # Load model
    model = torchvision.models.detection.retinanet_resnet50_fpn()
    model.load_state_dict(torch.load(model_path))
    print(f"Loading Model: {model_path}")
    model.to(device)

    # Load COCO-style dataset
    validation_sets = []
    for validation_dir in datasets:
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_sets.append(temp_validation)

    validation_set = ConcatDataset(validation_sets)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True, collate_fn=lambda x: (zip(*x)))

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_title)
    mlflow.log_param("Model", model_path)
    mlflow.log_param("Dataset", datasets)
    mlflow.end_run()
    with mlflow.start_run():
        for index, path in tqdm(enumerate(database)):
            # Load COCO-style dataset
            results = evaluate_over_time(model, path, index,validation_loader, evaluation_metrics, device)
            mlflow.log_metrics(results, index)
            results["date"] = database.path_to_date[path]
            database.add_metric(results)
            database.save()

            # Optimizer
        database.plot_all_metrics()
        mlflow.end_run()

def evaluate_over_time(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
    total_targets= []
    total_predictions = []
    metrics = {}

    model.eval()
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        images = torch.stack(images, dim=0)

        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        # total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        # total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

        if epoch%10 == 0 and plot:
            plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
            # plot=False

    for metric in evaluation_metrics:
        results = metric(total_predictions, total_targets)
        metrics.update(results)
    return metrics

def perform_timewise_benchmark(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5):
    database = Dataset_study(output_directory, datasets, experiment_title)
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
    model = Sentinel(normalize=False)
    model = model.load_original_model(model_path)
    model.to(device)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_title)
    mlflow.log_param("Model", model_path)
    mlflow.log_param("Dataset", datasets)
    mlflow.end_run()
    with mlflow.start_run():
        for index, path in tqdm(enumerate(database)):
            # Load COCO-style dataset
            validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
            validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True, collate_fn=lambda x: (zip(*x)))
            results = evaluate_over_time(model, path, index,validation_loader, evaluation_metrics, device)
            mlflow.log_metrics(results, index)
            results["date"] = database.path_to_date[path]
            database.add_metric(results)
            database.save()

            # Optimizer
        database.plot_all_metrics()
        mlflow.end_run()


if __name__ == "__main__":
    LMNT01_Datasets = ["/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-13_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-09-25_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-15_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-23_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-15_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-11-26_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-17_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-20_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2024-12-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-01-23_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-03_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-16_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT01Sat-2025-05-25_Channel_Mixture_C_Eval"]

    LMNT02_Datasets = ["/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-06_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-09-14_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-08_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-10_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-30_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-10-31_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-14_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-11-19_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-07_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-24_Channel_Mixture_C_Eval", "/data/Sentinel_Datasets/Finalized_datasets/LMNT02Sat-2024-12-31_Channel_Mixture_C_Eval"]
    
    RME04_Datasets = ["/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-07-19_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-09_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-04_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-11_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-13_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-08_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-06_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-07_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2025-06-04_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-15_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-28_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-12_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-05_Channel_Mixture_C", "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-07-09_Channel_Mixture_C"]

    LMNT01model = "/data/Sentinel_Datasets/Best_models/LMNT01_Base_model.pt"
    LMNT02model = "/data/Sentinel_Datasets/Best_models/LMNT02_Base_model.pt"
    RME04model = "/data/Sentinel_Datasets/Best_models/RME04_Base_model.pt"
    title = "Performance_over_time_LMNT01_on_LMNT02_data"
    output_dir = os.path.join("/data/Sentinel_Datasets/Experiments", title)
    metrics = [centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference]

    perform_timewise_benchmark(LMNT02_Datasets, LMNT01model, output_dir, title, metrics, device_no=3)
    perform_cumulative_evaluation(RME04_Datasets, RME04model, output_dir, title, metrics, device_no=5)

