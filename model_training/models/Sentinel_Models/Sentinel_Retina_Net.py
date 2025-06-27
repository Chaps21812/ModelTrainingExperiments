import torch
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from models.Subcomponents.NMS import NMS
from models.Subcomponents.Post_processing_adapters import RetinaToSentinel
from typing import Optional, List, Dict



class Sentinel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.retina_net = retinanet_resnet50_fpn()
        self.output_formatter = RetinaToSentinel()
        self.nms = NMS()

    def forward(self, images:list[torch.Tensor], targets:Optional[List[Dict[str,torch.Tensor]]]=None ) -> torch.Tensor:
        if self.training and not torch.jit.is_scripting():
            return self.retina_net(images, targets)
        
        # In eval mode: run inference and postprocess
        _, raw_outputs = self.retina_net(images)
        if not isinstance(raw_outputs, list):
            raw_outputs = [raw_outputs]
        tranformed_outputs: torch.Tensor = self.output_formatter.forward(raw_outputs)
        outputs: torch.Tensor = self.nms.forward(tranformed_outputs)

        return outputs

    @torch.jit.ignore()
    def load_original_model(self, model_path) -> None:
        self.retina_net.load_state_dict(torch.load(model_path))
        print(f"Loading Model: {model_path}")

    

