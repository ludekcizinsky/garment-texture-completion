from PIL import Image
import numpy as np
import os
import sys
import random
import torch.utils.data as data
from torchvision import transforms
from utils.data_utils import generate_random_polygon, generate_polygon_mask, apply_random_mask, generate_random_square_mask

class DatasetBase(data.Dataset):
    def __init__(self, max_v=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        return

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

class DatasetInpainting(DatasetBase):
    def __init__(self, image_folder, res=1024, mask_ratio=0.9, is_train=True, debug=False):
        super(DatasetInpainting, self).__init__()
        self._name = 'images'

        self.IMG_NORM_MEAN = 0.5
        self.IMG_NORM_STD = 0.5

        self.root = image_folder
        self.res = res
        self.mask_ratio = mask_ratio
        self.is_train = is_train
        self.debug = debug
        
        # Read dataset
        self._train_files()
        self._dataset_size = len(self.samples)
        
    def __len__(self):
        return self._dataset_size
    
    def _train_files(self,):
        samples = [] # sorted([os.path.join(self.root, c) for c in os.listdir(self.root)], key=str.lower)#[:-5] # []
        categories = sorted([c for c in os.listdir(self.root)], key=str.lower)
        for c in categories:
            category_items = sorted([
                os.path.join(self.root, c, s, file)
                for s in os.listdir(os.path.join(self.root, c))
                for file in ['partial_color.png', 'partial_color_shirt.png']
            ])
            samples.extend(category_items)  # -1
        # print(samples[:10])
        print('len(samples): ', len(samples))
        # sys.exit()
        if self.debug:
            self.samples = samples[:10]
        else:
            self.samples = samples
        
    def _process_image(self, image):
        image = image.resize((self.res, self.res), Image.Resampling.NEAREST)
        image = np.array(image).astype(np.float32) / 255  # [0, 1]
        image = 2 * image - 1  # [-1, 1]
        return image
    
    def generate_mask(self, shape, num_vertices, num_squares, min_square_size, max_square_size):
        # Generate random mask
        polygon = generate_random_polygon(shape, num_vertices)  # Generate a random polygon
        base_mask = generate_polygon_mask(shape, polygon)  # Create a polygon mask
        random_mask = apply_random_mask(base_mask, self.mask_ratio)  # Generate a random mask
        mask = generate_random_square_mask(random_mask, num_squares, min_square_size, max_square_size)  # Add square masks
        return mask

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        
        folder = os.path.dirname(self.samples[index])
        
        diffuse_path = os.path.join(folder, 'color.png')
        normal_path = os.path.join(folder, 'normal.png')
        roughness_path = os.path.join(folder, 'roughness.png')
        partial_path = self.samples[index]
        mask_path = self.samples[index].replace('partial_color', 'mask')
        
        diffuse_img = Image.open(diffuse_path).convert("RGB")
        if os.path.exists(normal_path):
            normal_img = Image.open(normal_path).convert("RGB")
            normal_img = self._process_image(normal_img)
            normal_img = np.transpose(normal_img, (2, 0, 1))
        else:
            normal_img = None
            
        if os.path.exists(roughness_path):
            roughness_img = Image.open(roughness_path).convert("RGB")
            roughness_img = self._process_image(roughness_img)
            roughness_img = np.transpose(roughness_img, (2, 0, 1))
        else:
            roughness_img = None
        
        if os.path.exists(partial_path):
            partial_img = Image.open(partial_path).convert("RGB")
            partial_img = self._process_image(partial_img)
            partial_img = np.transpose(partial_img, (2, 0, 1))
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask.resize((self.res, self.res), Image.Resampling.NEAREST)).astype(bool)
            mask = 2 * mask - 1  # [-1, 1]
        else:
            print(f"{partial_path} not exists! Generating mask")
            num_vertices, num_squares, min_square_size, max_square_size = 8, 5, 100, 200
            mask = self.generate_mask(diffuse_img.size[::-1], num_vertices, num_squares, min_square_size, max_square_size)
            diffuse_array = np.array(diffuse_img).astype(float)
            partial_img_array = diffuse_array * mask[..., None]
            partial_img = self._process_image(Image.fromarray(partial_img_array.astype(np.uint8)))
            mask = np.array(Image.fromarray(mask).resize((self.res, self.res), Image.Resampling.NEAREST)).astype(bool)
            mask = 2 * mask - 1  # [-1, 1]
            # (self.res, self.res, 3) -> (3, self.res, self.res)
            partial_img = np.transpose(partial_img, (2, 0, 1))
        
        # Resize + normalize -> (self.res, self.res, 3) [-1, 1]
        diffuse_img = self._process_image(diffuse_img)
        diffuse_img = np.transpose(diffuse_img, (2, 0, 1))
        
        if normal_img is None:
            normal_img = np.zeros((3, self.res, self.res))
        if roughness_img is None:
            roughness_img = np.zeros((3, self.res, self.res))
        
        # Pack data
        grid_data = {
            "prompt": "fill the missing parts of a fabric texture matching the existing colors and style",  # prompt,
            "partial_img": partial_img,
            "mask": mask,
            "full_diffuse_img": diffuse_img,
            "full_normal_img": normal_img,
            "full_roughness_img": roughness_img,
        }

        return grid_data
    
if __name__ == '__main__':
    dataset = DatasetInpainting('datasets/fabric_w_clean_logo_small_center')
    print(len(dataset))
    data = dataset[10]
    Image.fromarray(((data['mask'] + 1) / 2 * 255).astype(np.uint8)).save('test_output/mask.png')
    Image.fromarray(((data['partial_img'] + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)).save('test_output/partial_img.png')
    Image.fromarray(((data['full_diffuse_img'] + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)).save('test_output/full_diffuse_img.png')
    Image.fromarray(((data['full_normal_img'] + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)).save('test_output/full_normal_img.png')
    Image.fromarray(((data['full_roughness_img'] + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)).save('test_output/full_roughness_img.png')