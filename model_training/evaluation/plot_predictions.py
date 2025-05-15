import os
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import mlflow
from preprocessing_techniques.preprocessing import zscale
import torch

#Plotting predictions and calculating precision scores
def plot_prediction_bbox(images, predictions, targets, output_dir, epoch):
    os.makedirs(os.path.join(output_dir, "evaluation_images"), exist_ok=True)
    for index, (image, prediction_results, target_results) in enumerate(zip(images, predictions, targets)):
        image_path = os.path.join(output_dir, "evaluation_images", f"prediction_S{index}_E{epoch}.png")
        target_boxes = target_results["boxes"].detach().cpu()
        id = target_results["image_id"].detach().cpu()
        prediction_boxes = prediction_results["boxes"].detach().cpu()
        prediction_scores = prediction_results["scores"].detach().cpu()

        zscaled_image = zscale(image[0,:,:].detach().cpu().numpy())[0,:,:]

        fig, ax = plt.subplots()
        plt.title(f"{id} Predictions and Targets S{index} E{epoch}")
        img_hwc = np.transpose(image.detach().cpu(), (1, 2, 0))
        plt.imshow(img_hwc)
        for line_index,line_row in enumerate(target_boxes):
            x1 = line_row[0].item()
            y1 = line_row[1].item()
            x2 = line_row[2].item()
            y2 = line_row[3].item()
            x = (x2+x1)/2
            y = (y2+y1)/2
            w = (x2-x1)
            h = (y2-y1)
            square = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='green', facecolor='none', alpha=.3)
            ax.add_patch(square)
            ax.plot(x, y, 'o', alpha=.3, color="green")
        for line_index,line_row in enumerate(prediction_boxes):
            x1 = line_row[0].item()
            y1 = line_row[1].item()
            x2 = line_row[2].item()
            y2 = line_row[3].item()
            x = (x2+x1)/2
            y = (y2+y1)/2
            w = (x2-x1)
            h = (y2-y1)
            square = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none', alpha=.3)
            plt.text(x,y,str(prediction_scores[line_index].item()), alpha=.3, color="red")
            ax.add_patch(square)
            ax.plot(x, y, 'o', alpha=.3, color="red")
        plt.savefig(image_path)
        plt.close()
        torch.cuda.empty_cache()
        mlflow.log_artifact(image_path)

def plot_prediction_bbox_annotation(images, predictions, targets, output_dir, epoch, padding = 30):
    os.makedirs(os.path.join(output_dir, "evaluation_images"), exist_ok=True)
    for index, (image, prediction_results, target_results) in enumerate(zip(images, predictions, targets)):
        target_boxes = target_results["boxes"].detach().cpu()
        id = target_results["image_id"].detach().cpu()
        prediction_boxes = prediction_results["boxes"].detach().cpu()
        prediction_scores = prediction_results["scores"].detach().cpu()

            # Convert to HWC for matplotlib

        fig, ax = plt.subplots()
        plt.title(f"{id} Predictions and Targets S{index} E{epoch}")
        img_hwc = np.transpose(image.detach().cpu(), (1, 2, 0))
        plt.imshow(img_hwc)
        for line_index,line_row in enumerate(target_boxes):
            image_path = os.path.join(output_dir, "evaluation_images", f"prediction_S{index}_E{epoch}_A{line_index}.png")
            x1 = line_row[0].item()
            y1 = line_row[1].item()
            x2 = line_row[2].item()
            y2 = line_row[3].item()
            x = (x2+x1)/2
            y = (y2+y1)/2
            w = (x2-x1)
            h = (y2-y1)
            square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='green', facecolor='none', alpha=.9)
            ax.add_patch(square)
            ax.plot(x, y, 'o', alpha=.9, color="green")
            plt.xlim(x1-padding, x2+padding)
            plt.ylim(y1-padding, y2+padding)
            for prediction_index,prediction_row in enumerate(prediction_boxes):
                x1 = prediction_row[0].item()
                y1 = prediction_row[1].item()
                x2 = prediction_row[2].item()
                y2 = prediction_row[3].item()
                x = (x2+x1)/2
                y = (y2+y1)/2
                w = (x2-x1)
                h = (y2-y1)
                square = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='r', facecolor='none', alpha=.3)
                plt.text(x,y,str(prediction_scores[prediction_index].item()), color="red", alpha=.3)
                ax.add_patch(square)
                ax.plot(x, y, 'ro', alpha=.3)
            plt.savefig(image_path)
            plt.close()
            torch.cuda.empty_cache()
            mlflow.log_artifact(image_path)
