from torchvision.datasets import CocoDetection
from torchvision import transforms
import torch
import numpy as np
import imageio.v3 as iio
from PIL import Image
import os
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import numpy as np
import imageio.v3 as iio
from pycocotools.coco import COCO

import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

def collate_fn(batch):
    images, targets, ids = zip(*batch)
    return list(images), list(targets)

class Coco16bitGray(VisionDataset):
    """COCO Detection Dataset Loader for 16-bit grayscale PNGs"""
    def __init__(
        self,
        data_folder: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            data_folder: Path to dataset folder containing 'images/' and 'annotations/annotations.json'
            transform: optional function to transform image tensor
            target_transform: optional function to transform annotations
            transforms: optional function taking (image, target) and returning transformed pair
        """
        img_folder = os.path.join(data_folder)
        ann_file = os.path.join(data_folder, "annotations", "annotations.json")
        super().__init__(img_folder, transforms, transform, target_transform)
        
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform

    def _load_image(self, id: int) -> torch.Tensor:
        """Load 16-bit grayscale image as float tensor [1,H,W] normalized to [0,1]"""
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = iio.imread(os.path.join(self.root, path))
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 65535.0
        # if img_tensor.ndimension() > 3:
        #     img_tensor = img_tensor[:,:,:,0]

        # # Convert to float and normalize
        # img_tensor = torch.from_numpy(img).float() / 65535.0  # [H, W]
        # # Add channel dimension and replicate to 3 channels
        # img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        return img_tensor

    def _load_target(self, id: int) -> List[Any]:
        """Load COCO annotations for image id"""
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be an integer, got {type(index)}")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target, id

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires `pycocotools <https://github.com/ppwwyyxx/cocoapi>`_ to be installed,
    which could be installed via ``pip install pycocotools`` or ``conda install conda-forge::pycocotools``.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, id

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    It requires `pycocotools <https://github.com/ppwwyyxx/cocoapi>`_ to be installed,
    which could be installed via ``pip install pycocotools`` or ``conda install conda-forge::pycocotools``.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.PILToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]
