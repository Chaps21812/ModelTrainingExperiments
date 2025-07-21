import torch

class RetinaToSentinel(torch.nn.Module):
    def __init__(self):
        super(RetinaToSentinel, self).__init__()

    def forward(self, outputs:list[dict[str, torch.Tensor]]) -> torch.Tensor:
        #[centroid x, centroid y, box height, box width, box confidence]
        #[BATCH_NUM, 5, NUM_BOXES]
        max_length = 0
        output_tensors = []
        for batch_num in outputs:
            storage_vector = torch.zeros((5, len(batch_num["scores"])))
            storage_vector[0,:] = (batch_num["boxes"][:,2]+batch_num["boxes"][:,0])/2
            storage_vector[1,:] = (batch_num["boxes"][:,3]+batch_num["boxes"][:,1])/2
            storage_vector[2,:] = (batch_num["boxes"][:,2]-batch_num["boxes"][:,0])
            storage_vector[3,:] = (batch_num["boxes"][:,3]-batch_num["boxes"][:,1])
            if batch_num["scores"].numel() == 0:
                storage_vector[4,:] = batch_num["scores"].view(0)
            elif batch_num["scores"].ndim == 1:
                storage_vector[4,:] = batch_num["scores"]
            else:
                storage_vector[4,:] = batch_num["scores"].transpose(0,1)
            max_length = max(max_length, len(batch_num["boxes"]))
            output_tensors.append(storage_vector)
        finalized_tensors = torch.zeros((len(output_tensors), 5, max_length))
        for i, tensor in enumerate(output_tensors):
            finalized_tensors[i, :, :tensor.shape[1]] = tensor
        return finalized_tensors
    
    @torch.jit.ignore()
    def convert_targets(self, outputs:list[dict[str, torch.Tensor]]) -> torch.Tensor:
        #[centroid x, centroid y, box height, box width, box confidence]
        #[BATCH_NUM, 5, NUM_BOXES]
        max_length = 0
        output_tensors = []
        for batch_num in outputs:
            storage_vector = torch.zeros((5, len(batch_num["labels"])))
            storage_vector[0,:] = (batch_num["boxes"][:,2]+batch_num["boxes"][:,0])/2
            storage_vector[1,:] = (batch_num["boxes"][:,3]+batch_num["boxes"][:,1])/2
            storage_vector[2,:] = (batch_num["boxes"][:,2]-batch_num["boxes"][:,0])
            storage_vector[3,:] = (batch_num["boxes"][:,3]-batch_num["boxes"][:,1])
            storage_vector[4,:] = torch.ones_like(storage_vector[4,:])
            max_length = max(max_length, len(batch_num["boxes"]))
            output_tensors.append(storage_vector)
        finalized_tensors = torch.zeros((len(output_tensors), 5, max_length))
        for i, tensor in enumerate(output_tensors):
            finalized_tensors[i, :, :tensor.shape[1]] = tensor
        return finalized_tensors

