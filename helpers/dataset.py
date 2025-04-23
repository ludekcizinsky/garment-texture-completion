from PIL import Image
import numpy as np
import os
import cv2
from torch import utils

class InpaintingDataset(utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.texture_paths = [
            os.path.join(cfg.data.pbr_maps_path, folder)
            for folder in os.listdir(cfg.data.pbr_maps_path)
        ]

        mask = np.array(Image.open(cfg.data.mask_path).convert("L")) # Load mask as grayscale
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST).astype("float32") / 255.0
        self.mask = np.stack([mask] * 3, axis=-1)  # Convert to 3 channels
 
    def __len__(self):
        return len(self.texture_paths)
            
    def _normalise_image(self, image):
        image = np.array(image) / 255  # [0, 1]
        image = 2 * image - 1  # [-1, 1]
        return image

    def _denormalise_image(self, image):
        image = (image + 1) / 2
        image = np.clip(image, 0, 1)
        return image
    
    def _channels_first(self, image):
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        elif len(image.shape) == 4:
            return np.transpose(image, (0, 3, 1, 2))
        else:
            raise ValueError("Image must be either 3D or 4D tensor")
    
    def _channels_last(self, image):
        if len(image.shape) == 3:
            return np.transpose(image, (1, 2, 0))
        elif len(image.shape) == 4:
            return np.transpose(image, (0, 2, 3, 1))
        else:
            raise ValueError("Image must be either 3D or 4D tensor")

    def _load_image_as_array(self, path):
        image = Image.open(path).convert("RGB")
        image = np.array(image).astype(np.float32)
        return image

    def __getitem__(self, index):

        diffuse_img = self._load_image_as_array(
            os.path.join(self.texture_paths[index], "texture_diffuse.png")
        )
        partial_img = diffuse_img * self.mask

        diffuse_img = self._normalise_image(diffuse_img)
        partial_img = self._normalise_image(partial_img)

        diffuse_img = self._channels_first(diffuse_img)
        partial_img = self._channels_first(partial_img)

        grid_data = {
            "partial_diffuse_img": partial_img,
            "full_diffuse_img": diffuse_img,
        }

        return grid_data
    
if __name__ == '__main__':
    pass