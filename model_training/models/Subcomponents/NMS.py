import torch



def NMS(predictions:torch.Tensor, IOU_Threshold=.25) -> torch.tensor:
    tensor_list = []
    max_detections = 0
    for batch in predictions:
        reduced_detections = _NMS(batch, IOU_Threshold)
        tensor_list.append(reduced_detections)
        max_detections = max(max_detections, reduced_detections.shape[0])
    results_tensor = torch.zeros((len(tensor_list),max_detections,6))
    for index, tensor in enumerate(tensor_list):
        results_tensor[index, :,:] = tensor

    return results_tensor


def _NMS(predictions:torch.Tensor, IOU_Threshold) -> torch.tensor:
    sorted_indicies = predictions[:,4].argsort(dim=1) #[detections]
    sorted_indicies = sorted_indicies.squeeze()

    eof = False
    index = 0
    while not eof:
        best_box = predictions[sorted_indicies[index], 0:4]
        if index >= len(predictions):
            eof=True
        index+=1

