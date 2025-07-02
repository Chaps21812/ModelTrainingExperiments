from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from .format_targets import format_targets_bboxes
from .image_stitching import partition_images, recombine_annotations, generate_crops
import models.LETR.src.util.misc as utils
import math
import sys
from typing import Iterable

def train_one_epoch(model, optimizer, dataloader:DataLoader, device, epoch):
    model.train()

    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return loss_dict

def train_image_stitching(model, optimizer, dataloader:DataLoader, device, epoch, sub_batch_size=40):
    model.train()

    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        images_cropped, targets_cropped = generate_crops(images, targets, device=device)

        sub_batch_image = [images_cropped[i:i + sub_batch_size] for i in range(0, len(images_cropped), sub_batch_size)]
        sub_batch_target = [targets_cropped[i:i + sub_batch_size] for i in range(0, len(targets_cropped), sub_batch_size)]

        for sub_images, sub_targets in zip(sub_batch_image, sub_batch_target):
            loss_dict = model(sub_images, sub_targets)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            torch.cuda.empty_cache() 

    return loss_dict

def train_LETR_one_epoch(model, criterion, postprocessors, dataloader, optimizer, device, epoch, max_norm, args):
    model.train()
    criterion.train()

    counter = 0
    torch.cuda.empty_cache()
    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, origin_indices = model(images, postprocessors, targets, criterion)
        loss_dict = criterion(outputs, targets, origin_indices)

            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v   for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]  for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    return loss_dict_reduced_scaled

def train_DEF_DETR_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    counter = 0
    torch.cuda.empty_cache()
    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

    return loss_dict_reduced_scaled