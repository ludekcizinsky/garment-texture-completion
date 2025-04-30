from PIL import Image
import numpy as np
import os
import cv2
from torch import utils
import torch
from torch.utils.data import IterableDataset, get_worker_info

from helpers import data_utils

def get_dataloaders(cfg):
    full_ds = InpaintingDataset(cfg)
    N = len(full_ds)

    all_idx = np.random.permutation(N)
    split   = int(cfg.data.trn_frac * N)
    train_idx, val_idx = all_idx[:split].tolist(), all_idx[split:].tolist()


    train_ds = ResumableInpaintingIterableDataset(cfg, indices=train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,     # must be False for IterableDataset
    )

    val_ds = ResumableInpaintingIterableDataset(cfg, indices=val_idx)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,     # must be False for IterableDataset
    )

    return train_loader, val_loader


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
    
class ResumableInpaintingIterableDataset(IterableDataset):
    def __init__(self, cfg, indices=None):
        super().__init__()
        self.dataset = InpaintingDataset(cfg)
        self.indices = indices if indices is not None else list(range(len(self.dataset)))
        # cursors for resume
        self.position = 0
        self.worker_positions = {}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # single‐process
            start, step, wid = self.position, 1, None
        else:
            wid         = worker_info.id
            n_workers   = worker_info.num_workers
            start       = self.worker_positions.get(wid, wid)
            step        = n_workers

        for idx_pos in range(start, len(self.indices), step):
            real_idx = self.indices[idx_pos]
            item     = self.dataset[real_idx]

            # advance cursor(s)
            if wid is None:
                self.position = idx_pos + 1
            else:
                self.worker_positions[wid] = idx_pos + step
                print(f"Worker {wid} position: {self.worker_positions[wid]}")

            yield item

    def state_dict(self):
        return {
            "position": self.position,
            "worker_positions": self.worker_positions,
        }

    def load_state_dict(self, state):
        self.position         = state.get("position", 0)
        self.worker_positions = state.get("worker_positions", {})
        print("------- Dataset state loaded -------")
        print(f"position: {self.position}")
        print(f"worker_positions: {self.worker_positions}")
 