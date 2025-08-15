import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import os
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import mlflow
from tqdm import tqdm
from evaluation.evaluation_metrics import centroid_accuracy, calculate_bbox_metrics, calculate_centroid_difference
from evaluation.plot_predictions import plot_prediction_bbox, plot_prediction_bbox_annotation


def evaluate(model, dataset_directory:str, epoch:int, dataloader:DataLoader, coco_data_set:COCO):
    predictions = []
    total_targets= []
    total_predictions = []

    img_ids = coco_data_set.getImgIds()
    diction = {}
    for id in img_ids:
        diction[id]=True
    temp_iterator = iter(dataloader)
    dummyID = None
    while dummyID is None:
        images, targets = next(temp_iterator)
        for item in targets:
            try: 
                if item[0]["image_id"] in diction:
                    dummyID = item[0]["image_id"]
            except: pass

    model.eval()
    plot = True
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets(targets, dummyID)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        total_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        total_predictions.extend([{k: v.detach().cpu() for k, v in t.items()} for t in outputs])


        if epoch%10 == 0 and plot:
            # plot_prediction_bbox(images, outputs, targets, dataset_directory, epoch)
            # plot_prediction_bbox_annotation(images, outputs, targets, dataset_directory, epoch)
            plot=False

        for i, output in enumerate(outputs):
            boxes = output['boxes'].tolist()
            scores = output['scores'].tolist()
            labels = output['labels'].tolist()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                image_id = targets[i]["image_id"].item()
                if image_id in diction:
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format
                        "score": float(score),
                    })
    centroid_metrics = centroid_accuracy(total_predictions, total_targets)
    average_precisionts = calculate_bbox_metrics(total_predictions, total_targets)
    centroid_distance_accuracy = calculate_centroid_difference(total_predictions, total_targets)
    mlflow.log_metrics(centroid_metrics, epoch)
    mlflow.log_metrics(average_precisionts, epoch)
    mlflow.log_metrics(centroid_distance_accuracy, epoch)



    # Save to JSON
    results_dir = os.path.join(dataset_directory, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"epoch_{epoch}_predictions.json")
    with open(results_file, "w") as f:
        json.dump(predictions, f)
    return results_file

def train_one_epoch(model, optimizer, dataloader:DataLoader, device, epoch ):
    model.train()

    temp_iterator = iter(dataloader)
    dummyID = None
    while dummyID is None:
        images, targets = next(temp_iterator)
        for item in targets:
            try: dummyID = item[0]["id"]
            except: pass

    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        dataloader
        targets = format_targets(targets, dummyID=dummyID)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return loss_dict


def format_targets(targets, dummyID):
    formatted_targets = []

    for image_annots in targets:  # Loop over each image
        boxes = []
        labels = []

        for obj in image_annots:  # Loop over objects in that image
            boxes.append(obj["bbox"])
            labels.append(int(obj["category_id"]))
        if len(image_annots)==0:
            target_dict = {
                "image_id": torch.tensor(dummyID),
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



if __name__ == "__main__":

    train_params = {
        "epochs": 500,
        "batch_size": 2,
        "lr": 4e-4, #sqrt(batch_size)*4e-4
        "model_path": None,
        "training_dir": "/mnt/c/Users/david.chaparro/Documents/data/RME04TestingSet/train",
        "validation_dir": "/mnt/c/Users/david.chaparro/Documents/data/RME04TestingSet/val",
        "gpu": 0
    }

    # Custom transforms (RetinaNet expects images and targets)
    transform = T.Compose([
        T.ToTensor(),
        # T.RandomHorizontalFlip(0.5),
        # T.Resize(1024),
    ])


    # Dataset paths
    training_dir = train_params["training_dir"]
    validation_dir = train_params["validation_dir"]
    base_dir = os.path.dirname(training_dir)
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # === Load Ground Truth ===
    coco_gt = COCO(os.path.join(validation_dir, "annotations", "annotations.json"))

    # Load COCO-style dataset
    training_set = CocoDetection(root=training_dir, annFile=os.path.join(training_dir, "annotations", "annotations.json"), transforms=transform)
    validation_set = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
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
    optimizer = torch.optim.SGD(params, lr=train_params['lr'], momentum=0.9, weight_decay=0.0005)

    # Training loop (simplified)

    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("RetinaNet_Training_Over_Time")
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_params(train_params)
        model.train()
        for epoch in range(train_params["epochs"]):
            losses = train_one_epoch(model, optimizer, training_loader, device, epoch)
            mlflow.log_metrics(losses, epoch) 
            predictions_path = evaluate(model, validation_dir, epoch, validation_loader, coco_gt)
            torch.save(model.state_dict(), os.path.join(models_dir,f"retinanet_weights_E{epoch}.pt"))
            # mlflow.log_metrics(metrics_dict, epoch) 
        mlflow.end_run()



