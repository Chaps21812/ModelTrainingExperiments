
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from models.Sentinel_Models.Sentinel_Retina_Net_Stitch import Sentinel_Panoptic
import torch
import torchvision
import os

model_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/PanopticL1E49.pt"
ts_name = os.path.basename(model_path).replace(".pt", ".torchscript")
save_path = os.path.dirname(model_path)

model = Sentinel_Panoptic()
model.eval()
model.load_original_model(model_path)
TS_Model = torch.jit.script(model)
path = os.path.join(save_path,ts_name )
TS_Model.save(path)
print("Saved Model")

model = torch.jit.load(str(path))  # cast to string if pathlib.Path
model.eval()  # Ensure it's in inference mode