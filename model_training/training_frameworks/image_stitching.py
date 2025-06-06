import torch

#Images is a list of tensors
#Targets is a list of dictionaries, with image_id, boxes: tensor[labels], and labels: tensor[N_boxes, 4]
def partition_images(images:list, targets:list=None, device="cuda:0", sizeX:int=512, sizeY:int=512) -> tuple[list,list]:
    cropped_images = []
    cropped_targets = []
    for j, image in enumerate(images):
        x_res = image.shape[2] #Watch for X,Y Switch here
        y_res = image.shape[1] #Watch for X,Y Switch here

        x_steps = (x_res//sizeX+1)*2
        y_steps = (y_res//sizeY+1)*2

        x_step_size = int(sizeX/2)
        y_step_size = int(sizeY/2)
        for x_step in range(x_steps+1):
            for y_step in range(y_steps+1):
                x_center = (x_step+1)*x_step_size
                y_center = (y_step+1)*y_step_size

                lower_x_bound = max(0, int(x_center-sizeX/2))
                upper_x_bound = min(x_res, int(x_center+sizeX/2))
                lower_y_bound = max(0, int(y_center-sizeY/2))
                upper_y_bound = min(y_res, int(y_center+sizeY/2))
                
                cropped_image = image[:,lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound] #Watch for X,Y Switch here
                if cropped_image.shape[1] ==0 or cropped_image.shape[2]==0:
                    continue
                standardized_image = torch.zeros([image.shape[0], sizeX, sizeY])
                standardized_image[:, 0:cropped_image.shape[1], 0:cropped_image.shape[2]] = cropped_image
    
                partition_dict = {} 
                partition_dict["image_index"] = j
                partition_dict["lower_x_bound"] = lower_x_bound
                partition_dict["lower_y_bound"] = lower_y_bound
                partition_dict["upper_x_bound"] = upper_x_bound
                partition_dict["upper_y_bound"] = upper_y_bound
                
                if targets is not None:
                    target = targets[j]
                    partition_dict["image_id"] = target["image_id"]
                    targets_tensors = torch.empty((0,4)).to(device)
                    labels_tensor = torch.empty((0,1)).to(device)
                    scores_tensor = torch.empty((0,1)).to(device)
                    for i, row in enumerate(target["boxes"]):
                        x_min = row[0]
                        y_min = row[1]
                        x_max = row[2]
                        y_max = row[3]

                        x_min_inbounds = lower_x_bound <= x_min <=upper_x_bound
                        x_max_inbounds = lower_x_bound <= x_max <=upper_x_bound
                        y_min_inbounds = lower_y_bound <= y_min <=upper_y_bound
                        y_max_inbounds = lower_y_bound <= y_max <=upper_y_bound
                        #This causes targets on the edge to be used
                        # contains_x = x_min_inbounds or x_max_inbounds
                        # contains_y = y_min_inbounds or y_max_inbounds
                        #This causes only targets that are completely in the frame to be used
                        contains_x = x_min_inbounds and x_max_inbounds
                        contains_y = y_min_inbounds and y_max_inbounds

                        x_on_edge = x_min_inbounds ^ x_max_inbounds
                        y_on_edge = y_min_inbounds ^ y_max_inbounds

                        if contains_x:
                            transformed_x_min = max(torch.tensor(lower_x_bound).to(device), x_min)-torch.tensor(lower_x_bound).to(device)
                            transformed_x_max = min(torch.tensor(upper_x_bound).to(device), x_max)-torch.tensor(lower_x_bound).to(device)
                        if contains_y: 
                            transformed_y_min = max(torch.tensor(lower_y_bound).to(device), y_min)-torch.tensor(lower_y_bound).to(device)
                            transformed_y_max = min(torch.tensor(upper_y_bound).to(device), y_max)-torch.tensor(lower_y_bound).to(device)
                        if contains_x and contains_y:
                            bounded_annotation = torch.tensor([transformed_x_min, transformed_y_min, transformed_x_max, transformed_y_max])
                            bounded_annotation = bounded_annotation.reshape((1,4)).to(device)
                            category_label = torch.tensor(int(target["labels"][i])).to(device)
                            score = torch.tensor([1]).to(device)
                            scores_tensor = torch.vstack([scores_tensor, score])
                            targets_tensors = torch.vstack([targets_tensors, bounded_annotation])
                            labels_tensor = torch.vstack([labels_tensor, category_label])
                    partition_dict["boxes"] = targets_tensors.to(device)
                    partition_dict["labels"] = labels_tensor.to(device)
                    partition_dict["scores"] = scores_tensor.to(device)

                cropped_targets.append(partition_dict)
                cropped_images.append(standardized_image)
    return cropped_images, cropped_targets

def recombine_annotations(empty_image_info, predictions, device="cuda:0"):
    image_compilation_dict = {}
    previous_bounds = []
    for target,prediction in zip(empty_image_info, predictions):
        #Create a dictionary containing all information relevant to cropped image
        bound_dict = {"LX":target["lower_x_bound"],
         "LY":target["lower_y_bound"],
         "UX":target["upper_x_bound"],
         "UY":target["upper_y_bound"], 
         "image_index":target["image_index"],
         "Detections": torch.empty((0,4)).to(device),
         "Scores": torch.empty((0,1)).to(device)}
        
        #If the image belongs to a certian original image, then do the appropriate math
        orig_index = bound_dict["image_index"]
        if orig_index not in image_compilation_dict:
            image_compilation_dict[orig_index] = {}
            image_compilation_dict[orig_index]["boxes"] = torch.empty((0,4)).to(device)
            image_compilation_dict[orig_index]["scores"] = torch.empty((0,1)).to(device)
            image_compilation_dict[orig_index]["labels"] = torch.empty((0,1)).to(device)
            previous_bounds = []

        #Look at all predictions for small cropping
        for l, row in enumerate(prediction["boxes"]):
            #Get data from cropping prediction
            score = prediction["scores"][l]
            label = prediction["labels"][l]
            _bbox = row.clone().detach()
            _bbox[0] += target["lower_x_bound"]
            _bbox[1] += target["lower_y_bound"]
            _bbox[2] += target["lower_x_bound"]
            _bbox[3] += target["lower_y_bound"]
            _bbox = _bbox.to(device)

            #Check if the bbox is completely inbounds for a previous detection
            completely_inbounds = False
            for bounded_area in previous_bounds:
                LX_inbounds = bounded_area["LX"] < _bbox[0] < bounded_area["UX"] 
                LY_inbounds = bounded_area["LY"] < _bbox[1] < bounded_area["UY"]
                UX_inbounds = bounded_area["LX"] < _bbox[2] < bounded_area["UX"]
                UY_inbounds = bounded_area["LY"] < _bbox[3] < bounded_area["UY"]
                completely_inbounds = LX_inbounds and LY_inbounds and UX_inbounds and UY_inbounds
                if completely_inbounds: break

            #If the bounding box hasnt been completely inbounds for a previous cropped image "new box", then add to detections
            if not completely_inbounds:
                image_compilation_dict[orig_index]["boxes"] = torch.vstack([image_compilation_dict[orig_index]["boxes"],_bbox]).to(device)
                image_compilation_dict[orig_index]["scores"] = torch.vstack([image_compilation_dict[orig_index]["scores"],score]).to(device)
                image_compilation_dict[orig_index]["labels"] = torch.vstack([image_compilation_dict[orig_index]["labels"],label]).to(device)

            #Else we have already seen this detection, do nothing. 
        #Add the previous bounds to the list of seen bounds to cross reference later
        previous_bounds.append(bound_dict)

    output_targets = []
    for i in range(len(image_compilation_dict)):
        output_targets.append(image_compilation_dict[i])
        
    return output_targets



if __name__ ==  "__main__":
    import torch
    from torchvision.datasets import CocoDetection
    import torchvision.transforms.v2 as T
    from torch.utils.data import DataLoader
    import os
    import tqdm
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from IPython.display import clear_output

    
    def plot_image_recombination(images, targets, padding = 20, show=False, alpha=.4):
        # for index, (image, prediction_results, target_results) in enumerate(zip(images, predictions, targets)):
        for index, (image, target_results) in enumerate(zip(images, targets)):
            target_boxes = target_results["boxes"].detach().cpu()
            # prediction_boxes = prediction_results["boxes"].detach().cpu()
            # prediction_scores = prediction_results["scores"].detach().cpu()

            img_hwc = np.transpose(image.detach().cpu(), (1, 2, 0))
            if len(target_boxes)>0:
                for line_index,line_row in enumerate(target_boxes):
                    plt.clf() 
                    plt.close('all')
                    fig, ax = plt.subplots(1,2, figsize=(10,4))
                    ax[0].set_title(f"{index} Full cropped image TGTs: {target_boxes.shape[0]}")
                    ax[0].imshow(img_hwc)
                    ax[1].set_title(f"{index} Target {line_index+1}")
                    ax[1].imshow(img_hwc)

                    x1 = line_row[0].item()
                    y1 = line_row[1].item()
                    x2 = line_row[2].item()
                    y2 = line_row[3].item()
                    x = (x2+x1)/2
                    y = (y2+y1)/2
                    w = (x2-x1)
                    h = (y2-y1)
                    square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=alpha)
                    ax[0].add_patch(square)
                    ax[0].plot(x, y, 'o', alpha=alpha, color="red")
                    square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=alpha)
                    ax[1].add_patch(square)
                    ax[1].plot(x, y, 'o', alpha=alpha, color="red")
                    ax[1].set_xlim(x1-padding, x2+padding)
                    ax[1].set_ylim(y1-padding, y2+padding)
                    if show:
                        plt.show()
                        # input("Press enter to continue")
                        plt.clf()   # Clears the current figure
                        plt.cla()   # Clears the current axes (optional)
                        plt.close()
                        clear_output(wait=True)
            else:
                plt.clf() 
                plt.close('all')
                fig, ax = plt.subplots(1,2, figsize=(10,4))
                ax[0].set_title(f"{index} Full cropped image TGTs: {target_boxes.shape[0]}")
                ax[0].imshow(img_hwc)
                if show:
                    plt.show()
                    # input("Press enter to continue")
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)

                plt.close()
                torch.cuda.empty_cache()
        plt.close()

    # def plot_image_stitch_bbox(images, predictions, targets, output_dir, epoch, padding = 30, show=False):
    def plot_image_stitch_bbox(images, targets, padding = 20, show=False, plot_empty=False, alpha=.4):
        # for index, (image, prediction_results, target_results) in enumerate(zip(images, predictions, targets)):
        for index, (image, target_results) in enumerate(zip(images, targets)):
            target_boxes = target_results["boxes"].detach().cpu()
            id = target_results["image_id"].detach().cpu()
            x_index = target_results["lower_x_bound"]
            y_index = target_results["lower_y_bound"]
            # prediction_boxes = prediction_results["boxes"].detach().cpu()
            # prediction_scores = prediction_results["scores"].detach().cpu()

            img_hwc = np.transpose(image.detach().cpu(), (1, 2, 0))
            if len(target_boxes)>0:
                for line_index,line_row in enumerate(target_boxes):
                    plt.clf() 
                    plt.close('all')
                    fig, ax = plt.subplots(1,2, figsize=(10,4))
                    ax[0].set_title(f"{id} Full cropped image X:{x_index} Y:{y_index} TGT: {target_boxes.shape[0]}")
                    ax[0].imshow(img_hwc)
                    ax[1].set_title(f"{id} Target {line_index+1}")
                    ax[1].imshow(img_hwc)

                    x1 = line_row[0].item()
                    y1 = line_row[1].item()
                    x2 = line_row[2].item()
                    y2 = line_row[3].item()
                    x = (x2+x1)/2
                    y = (y2+y1)/2
                    w = (x2-x1)
                    h = (y2-y1)
                    square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=alpha)
                    ax[0].add_patch(square)
                    ax[0].plot(x, y, 'o', alpha=alpha, color="red")
                    square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=alpha)
                    ax[1].add_patch(square)
                    ax[1].plot(x, y, 'o', alpha=alpha, color="red")
                    ax[1].set_xlim(x1-padding, x2+padding)
                    ax[1].set_ylim(y1-padding, y2+padding)
                    if show:
                        plt.show()
                        # input("Press enter to continue")
                        plt.clf()   # Clears the current figure
                        plt.cla()   # Clears the current axes (optional)
                        plt.close()
                        clear_output(wait=True)
            elif plot_empty:
                plt.clf() 
                plt.close('all')
                fig, ax = plt.subplots(1,2, figsize=(10,4))
                ax[0].set_title(f"{id} Full cropped image X:{x_index} Y:{y_index} TGT: {target_boxes.shape[0]}")
                ax[0].imshow(img_hwc)
                if show:
                    plt.show()
                    # input("Press enter to continue")
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)


                plt.close()
                torch.cuda.empty_cache()
        plt.close()

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

    
    train_params = {
        "epochs": 100,
        "batch_size": 4,
        "lr": 4e-4, #sqrt(batch_size)*4e-4
        "model_path": None,
        "training_dir": "/mnt/c/Users/david.chaparro/Documents/data/RME04TestingSet/train",
        "validation_dir": "/mnt/c/Users/david.chaparro/Documents/data/RME04TestingSet/val",
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
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

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
            images_cropped, targets_cropped = partition_images(images, targets, device=device)
            images_cropped_predicted, empty_targets = partition_images(images, device=device)
            # input("Look at image stitching")
            # plot_image_stitch_bbox(images_cropped, targets_cropped, show=True)
            # input("Look at images with ground truth")
            recombined_targets = recombine_annotations(targets_cropped, targets_cropped, device=device)
            # plot_image_recombination(images, recombined_targets, show=True)
            # input("Now look at images with no ground truth")
            recombined_targets_predicted = recombine_annotations(empty_targets, targets_cropped, device=device)
            # plot_image_recombination(images, recombined_targets_predicted, show=True)

            print("HI")            




