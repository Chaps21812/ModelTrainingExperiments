
from models.Sentinel_Models.Sentinel_Retina_Net import Sentinel
from models.Sentinel_Models.Sentinel_Retina_Net_Stitch import Sentinel_Panoptic
from models.Sentinel_Models.Sentinel_Retina_Net_NMS import SentinelNMS
import torch
import torchvision
import os

# model_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_LargeTrainDemo/models/RetinaNet-Large-Dataset_2025-10-06_04:47/RetinaNet-Large-Dataset_weights_E53.pt"
# model_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04_2025/models/RetinaNet-Large-Dataset_2025-10-08_03:11/RetinaNet-Large-Dataset_weights_E30.pt"
# model_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME01_2025/models/RetinaNet-RME01-2025_2025-10-07_11:11/RetinaNet-RME01-2025_weights_E25.pt"


LMNT01Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_LargeTrainDemo/models/RetinaNet-Large-Dataset_2025-10-06_04:47/RetinaNet-Large-Dataset_weights_E53.pt"
RME042025Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME04_2025/models/RetinaNet-Large-Dataset_2025-10-08_03:11/RetinaNet-Large-Dataset_weights_E30.pt"
RME012025Model = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/RME01_2025/models/RetinaNet-RME01-2025_2025-10-07_11:11/RetinaNet-RME01-2025_weights_E25.pt"
LMNT01_E249 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/LMNT01_E249.pt"
LMNT02_180="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/LMNT02_E180.pt"
RME04_249="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Best_models/RME04_E249.pt"

for model_path in [LMNT01Model,RME042025Model,RME012025Model,LMNT01_E249,LMNT02_180,RME04_249]:
    ts_name = os.path.basename(model_path).replace(".pt", "recentroid.torchscript")
    save_path = os.path.dirname(model_path)

    model = SentinelNMS(tc=0.0, recentroid=True)
    model.eval()
    model.load_original_model(model_path)
    TS_Model = torch.jit.script(model)
    path = os.path.join(save_path,ts_name )
    print(path)
    TS_Model.save(path)
    print("Saved Model")

    model = torch.jit.load(str(path))  # cast to string if pathlib.Path
    model.eval()  # Ensure it's in inference mode


#/home/davidchaparro/Repos/sentinelsatellitedetection/src/model/weights/