import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import os
import mlflow
from tqdm import tqdm
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from evaluation.evaluation_metrics import calculate_pr_curves, centroid_l2_accuracy, centroid_l2_accuracy_Per_Image
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from evaluation.dataset_studies import Dataset_study
from training_frameworks.format_targets import format_targets_bboxes
import torch.nn.functional as F

def evaluate_over_time(model, dataset_directory:str, epoch:int, dataloader:DataLoader, evaluation_metrics:list, device:str, plot:bool=False):
    total_targets= []
    total_predictions = []
    max_prediction_length = 0
    max_target_length = 0
    metrics = {}
    H=1
    W=1

    SNRS = []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)
            H=images[0].shape[1]
            W=images[0].shape[2]

            SNRS.extend([sum(annot["local_snr"] for annot in target)/len(target) if len(target) > 0 else 0.0 for target in targets])
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

    concurrent_tps = centroid_l2_accuracy_Per_Image(total_preds, total_tgts)
    
    metrics.update(concurrent_tps)
    metrics["SNRs"] = SNRS

    for metric in evaluation_metrics:
        results = metric(total_preds, total_tgts, image_height=(W,H))
        metrics.update(results)
    return metrics

def perform_timewise_benchmark(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5, database=None):
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
    model = Sentinel(normalize=False)
    model.load_original_model(model_path)
    model.to(device)
    model.eval()



    mlflow.log_param("Model", model_path)
    mlflow.log_param("Dataset", datasets)

    for index, path in tqdm(enumerate(database)):
        # Load COCO-style dataset
        validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
        validation_loader = DataLoader(validation_set, batch_size=20, shuffle=True, collate_fn=lambda x: (zip(*x)))
        results = evaluate_over_time(model, path, index,validation_loader, evaluation_metrics, device)
        mlflow.log_metrics(results, index)
        results["date"] = database.path_to_date[path]
        database.add_metric(results)
        database.save()

        # Optimizer
    database.plot_all_metrics()
    return database

def perform_cumulative_evaluation(datasets:list, model_path:str, output_directory:str, experiment_title, evaluation_metrics, device_no=5, database=None):
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
    model = Sentinel(normalize=False)
    model.load_original_model(model_path)
    model.to(device)
    model.eval()

    # Load COCO-style dataset
    validation_sets = []
    for validation_dir in datasets:
        temp_validation = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
        validation_sets.append(temp_validation)

    validation_set = ConcatDataset(validation_sets)
    validation_loader = DataLoader(validation_set, batch_size=20, shuffle=True, collate_fn=lambda x: (zip(*x)))

    results = evaluate_over_time(model, "", 0,validation_loader, evaluation_metrics, device)
    database.cumulative_metrics = results
    results["date"] = database.path_to_date[validation_dir]
    database.add_metric(results)
    database.save()
    mlflow.log_params(results)


    return database

if __name__ == "__main__":
    #Pull data from these variables
    R1Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_R1.pt"
    R2Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_R2.pt"
    R5Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_R5.pt"
    SR1Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_SR.pt"
    SR2Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_SR2_check.pt"
    SR5Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/CurriculumLearning_models/Curriculum_SR5.pt"

    TelescopeATestSet = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L1_Test"
    TelescopeBTestSet = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_L2_Test"
    TelescopecTestSet = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/Curriculum_R4_Test"

    #Adjust this code only
    run_name = f"S5R5_on_TA_Data{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
    model = SR5Model
    dataset = TelescopecTestSet
    device_no = 6
    #Adjust this code only

    metrics = [calculate_pr_curves, centroid_l2_accuracy]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Curriculum Learning Evaluation")
    mlflow.end_run()
    with mlflow.start_run(run_name=run_name):
        output_dir = os.path.join("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Experiments", run_name)
        database = perform_cumulative_evaluation([dataset], model, output_dir, run_name, metrics, device_no=device_no)
    mlflow.end_run()