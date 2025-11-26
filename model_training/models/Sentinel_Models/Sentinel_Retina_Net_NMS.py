import torch
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from models.Subcomponents.NMS import NMS
from models.Subcomponents.Post_processing_adapters import RetinaToSentinel
from typing import Optional, List, Dict
import numpy as np
from astropy.visualization import ZScaleInterval



class SentinelNMS(torch.nn.Module):
    def __init__(self, normalize:bool=True, tc=0.0,min_size=512, max_size=10000,recentroid=False ):
        super().__init__()
        self.retina_net = retinanet_resnet50_fpn( min_size=min_size, max_size=max_size)
        self.output_formatter = RetinaToSentinel()
        self.nms = NMS()
        self.normalize_outputs = normalize
        self.tc = tc
        self.recentroid_output = recentroid

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

        post_processed_tc = self.confidence_threshold(post_processed)
        if self.recentroid_output:
            recentroided = self.recentroid(images,post_processed_tc)
            return recentroided
        else:
            return post_processed_tc



    @torch.jit.ignore()
    def load_original_model(self, model_path) -> None:
        self.retina_net.load_state_dict(torch.load(model_path))
        # print(f"Loading Model: {model_path}")

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
    
    def confidence_threshold(self, outputs:torch.Tensor) -> torch.Tensor:
        max_size = 0
        filtered = []
        for i in range(outputs.shape[0]):
            keep = outputs[i, 4, :] > self.tc
            filtered.append(outputs[i, :, keep]) 
            max_size = max(max_size,outputs[i, :, keep].shape[1])

        out = torch.zeros((outputs.size(0), 5, max_size))
        for i,image in enumerate(filtered): 
            out[i, :, :image.shape[1]] = image
        return out

    def recentroid(self, images:torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
        for image_index,predictions in enumerate(outputs):
            for pindex in range(predictions.shape[1]):
                xc = predictions[0,pindex]
                yc = predictions[1,pindex]
                w = predictions[2,pindex]
                h = predictions[3,pindex]
                xmin = int(xc-w/2+2)
                xmax = int(xc+w/2+2)
                ymin = int(yc-h/2+2)
                ymax = int(yc+h/2+2)
                subsection = images[image_index,:,ymin:ymax+1,xmin:xmax+1]
                if subsection.numel()<2:
                    continue
                subsection += 1
                median_pixel_value = torch.median(subsection)
                stdev = torch.std(subsection)
                subsection -= median_pixel_value+stdev
                subsection[subsection < 0] = 0
                
                y_num_sum = torch.tensor(0.0)
                x_num_sum = torch.tensor(0.0)
                y_den_sum = torch.tensor(0.0)
                x_den_sum = torch.tensor(0.0)
                for y_index,xrow in enumerate(torch.sum(subsection, dim=0)):
                    for x_index,pixel_value in enumerate(xrow):
                        # if pixel_value>median_pixel_value:
                        x_num_sum = x_num_sum + (pixel_value**3)*float(x_index)
                        y_num_sum = y_num_sum + (pixel_value**3)*float(y_index)
                        x_den_sum = x_den_sum + (pixel_value**3)
                        y_den_sum = y_den_sum + (pixel_value**3)
                new_xc = x_num_sum/x_den_sum
                new_yc = y_num_sum/y_den_sum

                if torch.isnan(new_xc) or torch.isnan(new_yc):
                    print(f"Nans detected{torch.isnan(new_xc)} {torch.isnan(new_yc)}")
                    continue
                
                outputs[image_index,0,pindex] = new_xc+torch.tensor(xmin)
                outputs[image_index,1,pindex] = new_yc+torch.tensor(ymin)
        return outputs
            
