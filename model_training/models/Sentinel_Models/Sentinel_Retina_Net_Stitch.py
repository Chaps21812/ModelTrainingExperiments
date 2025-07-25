import torch
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from models.Subcomponents.NMS import NMS
from models.Subcomponents.Post_processing_adapters import RetinaToSentinel
from training_frameworks.image_stitching import ImageStitching
from typing import Optional, List, Dict
import numpy as np
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import os
from matplotlib import patches

def plot_demo(image_tensor, path, image_predictions:torch.Tensor=None):
    fig, ax = plt.subplots()
    img_hwc = np.transpose(image_tensor.detach().cpu(), (1, 2, 0))
    plt.imshow(img_hwc)
    if  image_predictions is not None:
        for line_index,line_row in enumerate(image_predictions.transpose(1,0)):
            x = line_row[0].item()
            y = line_row[1].item()
            w = line_row[2].item()
            h = line_row[3].item()
            x1 = x-w/2
            y1 = y-h/2
            x2 = x+w/2
            y2 = y+h/2
            square = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='red', facecolor='none', alpha=.4)
            ax.add_patch(square)
            # ax.plot(x, y, '.', alpha=.4, color="red")
        if image_predictions.numel() ==0:
            plt.close()
            return
        else:
            plt.savefig(path, dpi=800, bbox_inches='tight')
            plt.close()
            return
    plt.savefig(path, dpi=800, bbox_inches='tight')
    plt.close()

def plot_demo_list_predictions(image_tensor, path, image_predictions:dict=None):
    fig, ax = plt.subplots()
    img_hwc = np.transpose(image_tensor.detach().cpu(), (1, 2, 0))
    plt.imshow(img_hwc)
    if  image_predictions is not None:
        for l, row in enumerate(image_predictions["boxes"]):
            _bbox = row.clone().detach().cpu()
            x1 = _bbox[0]
            y1 = _bbox[1]
            x2 = _bbox[2]
            y2 = _bbox[3]
            x = (x1+x2)/2
            y = (y1+y2)/2
            w = (x2-x1)
            h = (y2-y1)
            square = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='red', facecolor='none', alpha=.4)
            ax.add_patch(square)
            # ax.plot(x, y, '.', alpha=.4, color="red")
        if image_predictions["boxes"].numel() ==0:
            plt.close()
            return
        else:
            plt.savefig(path, dpi=800, bbox_inches='tight')
            plt.close()
            return
    plt.savefig(path, dpi=800, bbox_inches='tight')
    plt.close()

class Sentinel_Panoptic(torch.nn.Module):
    def __init__(self, normalize:bool=True, sub_batch_size:int=32, save_image:str=""):
        super().__init__()
        self.retina_net = retinanet_resnet50_fpn()
        self.output_formatter = RetinaToSentinel()
        self.nms = NMS()
        self.stitcher = ImageStitching()
        self.normalize_outputs = normalize
        self.sub_batch_size = sub_batch_size
        self.save_image = save_image

    def forward(self, images:torch.Tensor, targets:Optional[List[Dict[str,torch.Tensor]]]=None ) -> torch.Tensor:
        device = images.device
        preprocessed:List[torch.Tensor] = self.preprocess(images)
        cropped_images, targets = self.stitcher.generate_crops(preprocessed, device=device)
        resolutions:List[List[int]] = self.get_resolutions(cropped_images)

        sub_batch_image = [cropped_images[i:i + self.sub_batch_size] for i in range(0, len(cropped_images), self.sub_batch_size)]
        sub_batch_targets = [targets[i:i + self.sub_batch_size] for i in range(0, len(targets), self.sub_batch_size)]

        temporary_outputs = []
        with torch.no_grad():
            for sub_images in sub_batch_image:        
                # In eval mode: run inference and postprocess
                if torch.jit.is_scripting():
                    _, raw_outputs = self.retina_net(sub_images)
                else: 
                    raw_outputs = self.retina_net(sub_images)
                if not isinstance(raw_outputs, list):
                    raw_outputs = [raw_outputs]
                temporary_outputs.extend(raw_outputs)
                # del sub_images
                # torch.cuda.empty_cache() 

        recombined_outputs = self.stitcher.recombine_annotations(targets, temporary_outputs,  device=device)
        tranformed_outputs: torch.Tensor = self.output_formatter.forward(recombined_outputs)
        outputs: torch.Tensor = self.nms.forward(tranformed_outputs)

        if self.save_image != "":
            for jindex,(sub_image, target) in enumerate(zip(cropped_images, targets)):
                image_number = target["image_id"]
                sub_image_number = target["image_step_id"]
                sub_image_filename = os.path.join(self.save_image, f"I{image_number}_S{sub_image_number}.jpg")
                major_image_filename = os.path.join(self.save_image, f"I{image_number}.jpg")
                sub_image_filename_annotated = os.path.join(self.save_image, f"I{image_number}_S{sub_image_number}-A.jpg")
                major_image_filename_annotated = os.path.join(self.save_image, f"I{image_number}-A.jpg")

                plot_demo(sub_image, sub_image_filename)
                plot_demo(preprocessed[image_number], major_image_filename)
                plot_demo_list_predictions(sub_image, sub_image_filename_annotated, temporary_outputs[jindex])
                plot_demo(preprocessed[image_number], major_image_filename_annotated, outputs[target["image_id"],:,:])


        if self.normalize_outputs:
            post_processed: torch.Tensor = self.normalize(outputs, resolutions)
        else: 
            post_processed: torch.Tensor = outputs

        return post_processed

    @torch.jit.ignore()
    def load_original_model(self, model_path) -> None:
        self.retina_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loading Model: {model_path}")

    def preprocess(self, images:torch.Tensor) -> List[torch.Tensor]:
        device = images.device
        image_list = []
        for image in images:
            image_list.append(image.to(device))
        return image_list 

    def get_resolutions(self, images:torch.Tensor) -> List[List[int]]:
        resolution_list:List[List[int]] = []
        for image in images:
            resolution_list.append([image.shape[-1], image.shape[-2]])
        return resolution_list 

    def normalize(self, outputs:torch.Tensor, resolutions:List[List[int]]) -> torch.Tensor:
        for i in range(len(outputs)):
            outputs[i,0,:] = outputs[i,0,:]/resolutions[i][0]
            outputs[i,1,:] = outputs[i,1,:]/resolutions[i][1]
            outputs[i,2,:] = outputs[i,2,:]/resolutions[i][0]
            outputs[i,3,:] = outputs[i,3,:]/resolutions[i][1]
        return outputs
