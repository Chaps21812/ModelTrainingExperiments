import torch
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from models.Subcomponents.NMS import NMS
from models.Subcomponents.Post_processing_adapters import RetinaToSentinel
from training_frameworks.image_stitching import ImageStitching
from typing import Optional, List, Dict
import numpy as np
from astropy.visualization import ZScaleInterval

class Sentinel_Panoptic(torch.nn.Module):
    def __init__(self, normalize:bool=True, sub_batch_size:int=32):
        super().__init__()
        self.retina_net = retinanet_resnet50_fpn()
        self.output_formatter = RetinaToSentinel()
        self.nms = NMS()
        self.stitcher = ImageStitching()
        self.normalize_outputs = normalize
        self.sub_batch_size = sub_batch_size

    def forward(self, images:torch.Tensor, targets:Optional[List[Dict[str,torch.Tensor]]]=None ) -> torch.Tensor:
        device = images.device
        preprocessed:List[torch.Tensor] = self.preprocess(images)
        cropped_images, targets = self.stitcher.generate_crops(preprocessed, device=device)
        resolutions:List[List[int]] = self.get_resolutions(cropped_images)

        sub_batch_image = [cropped_images[i:i + self.sub_batch_size] for i in range(0, len(cropped_images), self.sub_batch_size)]

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
                del sub_images
                torch.cuda.empty_cache() 

        recombined_outputs = self.stitcher.recombine_annotations(targets, temporary_outputs,  device=device)
        tranformed_outputs: torch.Tensor = self.output_formatter.forward(recombined_outputs)
        outputs: torch.Tensor = self.nms.forward(tranformed_outputs)
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
