import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import os
import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_frameworks.image_stitching import partition_images, recombine_annotations_V2, generate_crops
from training_frameworks.format_targets import format_targets_bboxes
from evaluation.plot_predictions import plot_image_stitch_bbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

train_params = {
    "epochs": 100,
    "batch_size": 4,
    "lr": 4e-4, #sqrt(batch_size)*4e-4
    "model_path": None,
    "training_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-13_Channel_Mixture_C",
    "validation_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/RME04_MixtureC_Final/RME04Sat-2024-06-13_Channel_Mixture_C",
    "gpu": 0,
    "momentum": 0.9,
    "weight_decay": 0.0005
}
train_params = {
    "epochs": 100,
    "batch_size": 4,
    "lr": 4e-4, #sqrt(batch_size)*4e-4
    "model_path": None,
    "training_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/satsim_sats_dataset",
    "validation_dir": "/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data_finalized/satsim_sats_dataset",
    "gpu": 0,
    "momentum": 0.9,
    "weight_decay": 0.0005
}

# Custom transforms (RetinaNet expects images and targets)
transform = T.Compose([
    T.ToTensor(),
])

# Dataset paths
training_dir = train_params["training_dir"]
validation_dir = train_params["validation_dir"]
base_dir = os.path.dirname(training_dir)

# Load COCO-style dataset
training_set = CocoDetection(root=training_dir, annFile=os.path.join(training_dir, "annotations", "annotations.json"), transforms=transform)
validation_set = CocoDetection(root=validation_dir, annFile=os.path.join(validation_dir, "annotations", "annotations.json"), transforms=transform)
training_loader = DataLoader(training_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))
validation_loader = DataLoader(validation_set, batch_size=train_params["batch_size"], shuffle=True, collate_fn=lambda x: (zip(*x)))

# Optimizer
device = torch.device(f"cuda:{train_params["gpu"]}" if torch.cuda.is_available() else "cpu")

for epoch in range(train_params["epochs"]):
    for images, targets in tqdm(training_loader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images_cropped, targets_cropped = generate_crops(images, targets, device=device)
        print(len(images_cropped))
        input("These are how many images")
        plot_image_stitch_bbox(images_cropped, targets_cropped,show=True)

        print("HI")            
