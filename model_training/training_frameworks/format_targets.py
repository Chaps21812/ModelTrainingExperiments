import torch

def format_targets_bboxes(targets):
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