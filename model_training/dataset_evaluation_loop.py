import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import os
import mlflow
from tqdm import tqdm
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation
from evaluation.dataset_studies import Dataset_study


def format_targets(targets):
    formatted_targets = []

    for image_annots in targets:  # Loop over each image
        boxes = []
        labels = []

        for obj in image_annots:  # Loop over objects in that image
            boxes.append(obj["bbox"])
            labels.append(int(obj["category_id"]))
        if len(image_annots)==0:
            target_dict = {
                "image_id": torch.tensor(0),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }
        else:
            target_dict = {
                "image_id": torch.tensor(obj["image_id"]),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
            target_dict["boxes"][:,2:] += target_dict["boxes"][:,:2]

        formatted_targets.append(target_dict)
    return formatted_targets


def evaluate(model, dataset_directory:str, epoch:int, dataloader:DataLoader, device):
    total_targets= []
    total_predictions = []

    model.eval()
    plot = True
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])

        if epoch%10 == 0 and plot:
            plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
            plot=False

    centroid_metrics = centroid_accuracy(total_predictions, total_targets)
    average_precisions = calculate_bbox_metrics(total_predictions, total_targets)
    combined_dict = centroid_metrics | average_precisions
    return combined_dict

def perform_timewise_benchmark(datasets:list, model_path:str, output_directory:str, experiment_title, device_no=5):
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

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("RME04_Sentinel_Sensor_Degredation")
    mlflow.end_run()
    with mlflow.start_run():
        for index, path in tqdm(enumerate(database)):
            # Load COCO-style dataset
            validation_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=transform)
            validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True, collate_fn=lambda x: (zip(*x)))
            results = evaluate(model, path, index, validation_loader, device=device)
            mlflow.log_metrics(results, index)
            results["date"] = database.path_to_date[path]
            database.add_metric(results)
            database.save()

            # Optimizer
        database.plot_all_metrics()
        mlflow.end_run()


if __name__ == "__main__":
    datasets=["/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-05-30/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-06-15/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-06-25/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-07-15/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-07-31/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-08-19/",
              "/home/davidchaparro/Datasets/MultiChannelRME04Sats-2024-08-28/",
              ]
    model = "/home/davidchaparro/Datasets/MultiChannelRME04Sats_Train/models/BestModelE42.pt"
    title = "MultiChannelRME04_experiment"
    output_dir = f"/home/davidchaparro/Datasets/{title}"

    
    perform_timewise_benchmark(datasets,model,output_dir,title, device_no=6)
