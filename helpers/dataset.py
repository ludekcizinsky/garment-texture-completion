from PIL import Image
import numpy as np
import os
import cv2
from torch import utils
import torch
from torch.utils.data import IterableDataset, get_worker_info
from multiprocessing import Manager

from tqdm import tqdm

from helpers import data_utils

def get_dataloaders(cfg):
    full_ds = InpaintingDataset(cfg)
    N = len(full_ds)

    all_idx = np.random.permutation(N)
    split   = N - cfg.data.val_size
    train_idx, val_idx = all_idx[:split].tolist(), all_idx[split:].tolist()

    # Create a Manager + shared dict for worker positions
    manager = Manager()
    trn_shared_worker_positions = manager.dict()

    train_ds = ResumableInpaintingIterableDataset(
        cfg,
        indices=train_idx,
        shared_worker_positions=trn_shared_worker_positions
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,     # must be False for IterableDataset
    )

    val_ds = InpaintingDataset(cfg, selected_indices=val_idx)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    return train_loader, val_loader


class InpaintingDataset(utils.data.Dataset):
    def __init__(self, cfg, selected_indices=None):
        super().__init__()

        self.cfg = cfg
        self.res = self.cfg.data.res

        self.texture_paths = sorted([
            os.path.join(cfg.data.pbr_maps_path, folder)
            for folder in os.listdir(cfg.data.pbr_maps_path)
        ])
        # self.texture_paths = [path for path in tqdm(self.texture_paths, desc="Loading texture paths") if os.path.exists(os.path.join(path, "texture_diffuse.png"))]

        if selected_indices is not None:
            self.texture_paths = [self.texture_paths[i] for i in selected_indices]


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
        mask = data_utils.normalise_image(self.mask)

        diffuse_img = data_utils.channels_first(diffuse_img)
        partial_img = data_utils.channels_first(partial_img)
        mask = data_utils.channels_first(mask)

        name = self.texture_paths[index].split("/")[-1]

        grid_data = {
            "partial_diffuse_img": partial_img,
            "full_diffuse_img": diffuse_img,
            "name": name,
            "mask": mask
        }

        return grid_data
    
class ResumableInpaintingIterableDataset(IterableDataset):
    def __init__(self, cfg, indices=None, shared_worker_positions=None):
        super().__init__()
        self.dataset = InpaintingDataset(cfg)
        self.indices = indices if indices is not None else list(range(len(self.dataset)))
        self.dataset_length = len(self.indices)

        # Use the shared Manager dict if provided, else fall back to a local dict
        if shared_worker_positions is None:
            self.worker_positions = {}
        else:
            self.worker_positions = shared_worker_positions

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            wid, n_workers = 0, 1
        else:
            wid       = worker_info.id
            n_workers = worker_info.num_workers

        # Start from the last absolute position for this worker (or wid for a fresh start)
        pos = self.worker_positions.get(wid, wid)

        # Infinite loop: cycle through self.indices over and over
        while True:
            real_idx = self.indices[pos % self.dataset_length]
            item = self.dataset[real_idx]

            # advance by n_workers so we stagger among workers
            pos += n_workers
            self.worker_positions[wid] = pos     # writes back into the Manager dict

            yield item

    def state_dict(self):
        # Convert the (possibly proxy) dict into a regular dict for serialization
        return {"worker_positions": dict(self.worker_positions)}

    def load_state_dict(self, state):
        loaded = state.get("worker_positions", {})
        # assume self.worker_positions is still the Manager.dict proxy
        self.worker_positions.clear()
        self.worker_positions.update(loaded)
