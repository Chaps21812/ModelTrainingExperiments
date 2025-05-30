from torch.utils.data import DataLoader
from tqdm import tqdm
from .format_targets import format_targets_bboxes

def train_one_epoch(model, optimizer, dataloader:DataLoader, device, epoch):
    model.train()

    for images, targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        images = list(img.to(device) for img in images)
        dataloader
        targets = format_targets_bboxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return loss_dict