from PIL import Image
import numpy as np
import os
import cv2
from torch import utils
import torch

from helpers import data_utils

def get_dataloaders(cfg):
    dataset = InpaintingDataset(cfg)

    train_size = int(cfg.data.trn_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    return train_dataloader, val_dataloader


class InpaintingDataset(utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.res = self.cfg.data.res

        self.texture_paths = [
            os.path.join(cfg.data.pbr_maps_path, folder)
            for folder in os.listdir(cfg.data.pbr_maps_path)
        ]

        mask = np.array(Image.open(cfg.data.mask_path).convert("L")) # Load mask as grayscale
        mask = cv2.resize(mask, (self.res, self.res), interpolation=cv2.INTER_NEAREST).astype("float32") / 255.0
        self.mask = np.stack([mask] * 3, axis=-1)  # Convert to 3 channels
 
    def __len__(self):
        return len(self.texture_paths)
            
    def __getitem__(self, index):

        diffuse_img = data_utils.load_image_as_array(os.path.join(self.texture_paths[index], "texture_diffuse.png"))
        diffuse_img = cv2.resize(diffuse_img, (self.res, self.res), interpolation=cv2.INTER_AREA)

        partial_img = diffuse_img * self.mask

        diffuse_img = data_utils.normalise_image(diffuse_img)
        partial_img = data_utils.normalise_image(partial_img)

        diffuse_img = data_utils.channels_first(diffuse_img)
        partial_img = data_utils.channels_first(partial_img)

        grid_data = {
            "partial_diffuse_img": partial_img,
            "full_diffuse_img": diffuse_img,
        }

        return grid_data
    
if __name__ == '__main__':
    pass