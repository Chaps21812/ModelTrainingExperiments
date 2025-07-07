import torch
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from models.Subcomponents.NMS import NMS
from models.Subcomponents.Post_processing_adapters import RetinaToSentinel
from typing import Optional, List, Dict
import numpy as np
from astropy.visualization import ZScaleInterval



class Sentinel(torch.nn.Module):
    def __init__(self, normalize:bool=True):
        super().__init__()
        self.retina_net = retinanet_resnet50_fpn()
        self.output_formatter = RetinaToSentinel()
        self.nms = NMS()
        self.normalize_outputs = normalize

    def forward(self, images:torch.Tensor, targets:Optional[List[Dict[str,torch.Tensor]]]=None ) -> torch.Tensor:
        preprocessed:List[torch.Tensor] = self.preprocess(images)
        resolutions:List[List[int]] = self.get_resolutions(images)

        if self.training and not torch.jit.is_scripting():
            return self.retina_net(images, targets)
        
        # In eval mode: run inference and postprocess
        if torch.jit.is_scripting():
            _, raw_outputs = self.retina_net(preprocessed)
        else: 
            raw_outputs = self.retina_net(preprocessed)
        if not isinstance(raw_outputs, list):
            raw_outputs = [raw_outputs]
        tranformed_outputs: torch.Tensor = self.output_formatter.forward(raw_outputs)
        outputs: torch.Tensor = self.nms.forward(tranformed_outputs)
        if self.normalize_outputs:
            post_processed: torch.Tensor = self.normalize(outputs, resolutions)
        else: 
            post_processed: torch.Tensor = outputs


        return post_processed

    @torch.jit.ignore()
    def load_original_model(self, model_path) -> None:
        self.retina_net.load_state_dict(torch.load(model_path))
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
