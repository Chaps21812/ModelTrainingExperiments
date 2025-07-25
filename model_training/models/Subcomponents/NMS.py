import torch

class NMS(torch.nn.Module):
    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, outputs:torch.Tensor) -> torch.Tensor:
        #[centroid x, centroid y, box height, box width, box confidence]
        #[BATCH_NUM, 5, NUM_BOXES]
        return self.NMS(outputs)

    def NMS(self, predictions:torch.Tensor, IOU_Threshold:float=.10) -> torch.Tensor:
        tensor_list = []
        max_detections = 0
        for batch in predictions:
            reduced_detections = self._NMS(batch, IOU_Threshold)
            tensor_list.append(reduced_detections)
            max_detections = max(max_detections, reduced_detections.shape[1])
        results_tensor = torch.zeros((len(tensor_list),5, max_detections))
        for index, tensor in enumerate(tensor_list):
            results_tensor[index, :,:tensor.shape[1]] = tensor

        return results_tensor


    def _NMS(self, predictions:torch.Tensor, IOU_Threshold:float) -> torch.Tensor:

        eof = False
        index = 0
        while not eof:
            if predictions.shape[1] == 0: 
                index+=1
                if index >= predictions.shape[1]:
                    eof=True
                continue

            sorted_indicies = predictions[4,:].argsort(dim=0, descending=True) #[detections]
            sorted_indicies = sorted_indicies.squeeze()
            if sorted_indicies.ndim == 0: 
                index+=1
                if index >= predictions.shape[1]:
                    eof=True
                continue

            best_box = predictions[0:4, sorted_indicies[index]]
            if best_box.ndim == 1:
                best_box = best_box.unsqueeze(1)
            assert best_box.ndim == 2
    
            ious = self._calculate_IOU(predictions, best_box)
            keep_scores = (ious<IOU_Threshold).transpose(0,1).squeeze() #Investigate the math here, > seems to do better for some reason
            keep_scores[sorted_indicies[index]] = True
            non_zero_scores = (predictions[4,:]>0)
            keep_scores = keep_scores & non_zero_scores
            predictions = predictions[:,keep_scores]

            index+=1
            if index >= predictions.shape[1]:
                eof=True
        return predictions
            
    def _calculate_IOU(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xc_prediction, yc_prediction, width_prediction, height_prediction = predictions[0, :], predictions[1, :], predictions[2, :], predictions[3, :]
        xc_target, yc_target, width_target, height_target = targets[0, :], targets[1, :], targets[2, :], targets[3, :]

        x1_prediction = xc_prediction-width_prediction/2
        y1_prediction = yc_prediction-height_prediction/2
        x2_prediction = xc_prediction+width_prediction/2
        y2_prediction = yc_prediction+height_prediction/2

        x1_target = xc_target-width_target/2
        y1_target = yc_target-height_target/2
        x2_target = xc_target+width_target/2
        y2_target = yc_target+height_target/2

        intersect_x1 = torch.maximum(x1_prediction[:,None], x1_target)
        intersect_y1 = torch.maximum(y1_prediction[:,None], y1_target)
        intersect_x2 = torch.minimum(x2_prediction[:,None], x2_target)
        intersect_y2 = torch.minimum(y2_prediction[:,None], y2_target)

        intersect_width = (intersect_x2-intersect_x1).clamp(min=0)
        intersect_height = (intersect_y2-intersect_y1).clamp(min=0)
        intersect_area = intersect_width*intersect_height

        prediction_area = (x2_prediction-x1_prediction)*(y2_prediction-y1_prediction)
        target_area = (x2_target-x1_target)*(y2_target-y1_target)

        union_area = prediction_area[:,None] + target_area - intersect_area

        iou= intersect_area/(union_area + 1e-8)

        return iou