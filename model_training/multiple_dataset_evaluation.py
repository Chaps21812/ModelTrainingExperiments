import torch
from torchvision.models.detection import retinanet_resnet50_fpn

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
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }
        else:
            target_dict = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
            target_dict["boxes"][:,2:] += target_dict["boxes"][:,:2]

        formatted_targets.append(target_dict)
    return formatted_targets


def evaluate(model, dataset_directory:str, epoch:int, dataloader:DataLoader):
    predictions = []

    model.eval()
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        targets = format_targets(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        for i, output in enumerate(outputs):
            boxes = output['boxes'].tolist()
            scores = output['scores'].tolist()
            labels = output['labels'].tolist()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": targets[i]["image_id"].item(),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format
                    "score": float(score),
                })

    # Save to JSON
    results_dir = os.path.join(dataset_directory, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"epoch_{epoch}_predictions.json")
    with open(results_file, "w") as f:
        json.dump(predictions, f)
    return results_file

if __name__ == "__main__":
    datasets = []
    model_path = ""
    results = []


    model = retinanet_resnet50_fpn()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # or model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for path in datasets:
        # === Load Ground Truth ===
        coco_gt = COCO(os.path.join(path, "annotations", "annotations.json"))

        # Load COCO-style dataset
        training_set = CocoDetection(root=path, annFile=os.path.join(path, "annotations", "annotations.json"), transforms=None)
        training_loader = DataLoader(training_set, batch_size=2, shuffle=True, collate_fn=lambda x: (zip(*x)))

        evaluation_results_path = evaluate(model, )


