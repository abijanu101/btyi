import numpy as np
import torch
import torchvision
import PIL.Image

from typing import List, Tuple, Dict
import pathlib

class MOT17Dataset(torch.utils.data.Dataset):
    def __init__(self, bbox_data:dict, dir:pathlib.Path, augmentation_enabled:bool):
        self.bbox_data = bbox_data
        self.dir = dir

        f = [len(v) for v in bbox_data.values()]
        self.cf = np.cumsum(f)
        self.chlg = list(bbox_data.keys())

        self.pre = torchvision.transforms.Compose([
            *([
                torchvision.transforms.GaussianBlur(),
                torchvision.transforms.ColorJitter(),
            ] if augmentation_enabled else []),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return self.cf[-1]

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, Dict]:
        if idx < 0:
            idx += len(self)
            if idx < 0: raise IndexError()

        i = sum(self.cf <= idx)
        chlg = self.chlg[i]
        frame = idx - (self.cf[i-1] if i!=0 else 0)
        frame += 1      # the file names are One-Indexed

        path = self.dir / chlg / 'img1' / f'{frame:06d}.jpg'
        img = PIL.Image.open(str(path))
        tensor = self.pre(img)

        return tensor, self.bbox_data[chlg][frame]

def collate_fn(batch:List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    img, bboxes = zip(*batch)
    imgs = torch.stack(img).shape

    print(bboxes)
    
