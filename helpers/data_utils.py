import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


def generate_random_polygon(image_shape, num_vertices=6):
    """
    Generate a random polygon.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        num_vertices (int): Number of vertices for the polygon.

    Returns:
        polygon (np.ndarray): Array of polygon vertices.
    """
    height, width = image_shape
    points = np.zeros((num_vertices, 2), dtype=np.int32)
    for i in range(num_vertices):
        x = np.random.randint(0, width)  # Random x-coordinate
        y = np.random.randint(0, height)  # Random y-coordinate
        points[i] = [x, y]
    # Use convex hull to create a closed polygon
    polygon = cv2.convexHull(points)
    return polygon

def generate_polygon_mask(image_shape, polygon):
    """
    Generate a binary mask based on a given polygon.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        polygon (np.ndarray): Array of polygon vertices.

    Returns:
        mask (np.ndarray): Binary mask where the polygon area is 1, others are 0.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Fill the polygon area with 1
    cv2.fillPoly(mask, [polygon], 1)
    return mask

def generate_random_square_mask(base_mask, num_squares, min_square_size, max_square_size):
    """
    Add random square masks to an existing mask.

    Args:
        base_mask (np.ndarray): The base binary mask (values 0 or 1).
        num_squares (int): Number of random squares to add.
        max_square_size (int): Maximum size of each square.

    Returns:
        square_mask (np.ndarray): Mask with random squares added.
    """
    square_mask = base_mask.copy()
    height, width = base_mask.shape

    for _ in range(num_squares):
        # Randomly determine the square size and position
        square_size = np.random.randint(min_square_size, max_square_size)
        x_start = np.random.randint(0, width - square_size)
        y_start = np.random.randint(0, height - square_size)

        # Set the square region to 0
        square_mask[y_start:y_start + square_size, x_start:x_start + square_size] = 0

    return square_mask

def apply_random_mask(base_mask, mask_ratio):
    """
    Generate a random mask on top of a base mask.

    Args:
        base_mask (np.ndarray): Initial binary mask (values 0 or 1).
        mask_ratio (float): Ratio of pixels in the mask to be randomly set to 0.

    Returns:
        random_mask (np.ndarray): Modified mask with some pixels randomly set to 0.
    """
    random_mask = base_mask.copy()
    # Find indices of valid pixels (value == 1)
    valid_indices = np.where(base_mask == 1)
    num_valid = len(valid_indices[0])
    # Determine how many pixels to set to 0
    num_to_zero = int(num_valid * mask_ratio)
    # Randomly select pixels to set to 0
    zero_indices = np.random.choice(num_valid, num_to_zero, replace=False)
    random_mask[valid_indices[0][zero_indices], valid_indices[1][zero_indices]] = 0
    return random_mask

def modify_texture_images(color_path, normal_path, roughness_path, logo_path, output_dir, resize_range=(0.2, 1.2)):
    """
    Modify texture images by pasting a logo onto the color map and adjusting the normal and roughness maps.

    Args:
        color_path (str): Path to the color texture image.
        normal_path (str): Path to the normal map image.
        roughness_path (str): Path to the roughness map image.
        logo_path (str): Path to the logo image.
        output_dir (str): Directory to save the modified images.
    """
    # Load images
    color = Image.open(color_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")
    if os.path.exists(normal_path):
        normal_opengl = Image.open(normal_path).convert("RGB")
        normal_opengl = normal_opengl.resize((1024, 1024))
        normal_opengl = np.array(normal_opengl)
    else:
        normal_opengl = None
    if os.path.exists(roughness_path):
        roughness = Image.open(roughness_path).convert("RGB")
        roughness = roughness.resize((1024, 1024))
        roughness = np.array(roughness)
    else:
        roughness = None
    
    color = color.resize((1024, 1024))
    # Get dimensions of color and logo images
    color_width, color_height = color.size
    logo_width, logo_height = logo.size

    # Randomly scale the logo within the resize range
    scale_factor = random.uniform(*resize_range)

    # Adjust the scale if the resized logo exceeds color dimensions
    max_scale_factor = min(color_width / logo_width, color_height / logo_height)
    scale_factor = min(scale_factor, max_scale_factor)

    # Resize the logo
    new_logo_size = (int(logo_width * scale_factor), int(logo_height * scale_factor))
    logo = logo.resize(new_logo_size, Image.Resampling.LANCZOS)
    logo_width, logo_height = logo.size

    # Randomly select position to paste the logo
    x_pos = random.randint(0, color_width - logo_width)
    y_pos = random.randint(0, color_height - logo_height)

    # Paste the logo onto the color image
    color.paste(logo, (x_pos, y_pos), logo)
    # Save the modified images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    color.save(os.path.join(output_dir, "color.png"))

    # Get the alpha channel of the logo as a mask
    logo_alpha = np.array(logo.split()[3])  # Extract alpha channel
    logo_mask = (logo_alpha > 0).astype(np.uint8)  # Binary mask for logo region

    # Create coordinate grids for the logo region
    yy, xx = np.meshgrid(
        np.arange(logo_height), np.arange(logo_width), indexing="ij"
    )
    valid_coords = logo_mask[yy, xx].astype(bool)  # Only process valid pixels
    
    # Compute indices for the normal and roughness modifications
    global_y = y_pos + yy[valid_coords]
    global_x = x_pos + xx[valid_coords]
        
    if normal_opengl is not None:
        # Adjust normal values in the logo region
        normals = normal_opengl[global_y, global_x, :3]
        normals[:, 0] = int(0.5 * 255) # np.minimum(0, normals[:, 0])  # Adjust X component
        normals[:, 1] = int(0.5 * 255) # np.minimum(0, normals[:, 1])  # Adjust Y component
        normals[:, 2] = int(255) # np.maximum(255, normals[:, 2])  # Adjust Z component
        normal_opengl[global_y, global_x, :3] = normals
        normal_opengl = Image.fromarray(np.clip(normal_opengl, 0, 255).astype(np.uint8))
        normal_opengl.save(os.path.join(output_dir, "normal.png"))
        
    if roughness is not None:
        # Adjust roughness values in the logo region
        roughness[global_y, global_x] = 255 * 0.52
        roughness = Image.fromarray(np.clip(roughness, 0, 255).astype(np.uint8))
        roughness.save(os.path.join(output_dir, "roughness.png"))    
    
    print(f"Logo pasted and saved to {output_dir}")
    
def modify_texture_images_small_center(
    color_path, normal_path, roughness_path, logo_path, output_dir, logo_width_range=(30, 140), res=512
):
    """
    Modify texture images by pasting a scaled logo onto the color map and adjusting the normal and roughness maps.

    Args:
        color_path (str): Path to the color texture image.
        normal_path (str): Path to the normal map image.
        roughness_path (str): Path to the roughness map image.
        logo_path (str): Path to the logo image.
        output_dir (str): Directory to save the modified images.
        logo_width_range (tuple): Range of widths for resizing the logo (min, max).
    """
    # Load images
    color = Image.open(color_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")
    if os.path.exists(normal_path):
        normal_opengl = Image.open(normal_path).convert("RGB")
        normal_opengl = normal_opengl.resize((res, res))
        normal_opengl = np.array(normal_opengl)
    else:
        normal_opengl = None
    if os.path.exists(roughness_path):
        roughness = Image.open(roughness_path).convert("RGB")
        roughness = roughness.resize((res, res))
        roughness = np.array(roughness)
    else:
        roughness = None

    color = color.resize((res, res))
    color_width, color_height = color.size

    # Randomly scale the logo width and maintain aspect ratio
    random_logo_width = random.randint(*logo_width_range)
    scale_factor = random_logo_width / logo.width
    new_logo_size = (random_logo_width, int(logo.height * scale_factor))
    logo = logo.resize(new_logo_size, Image.Resampling.LANCZOS)
    logo_width, logo_height = logo.size

    # Define center point and radius for logo placement
    center_x = 0.5 * color_width
    center_y = 0.4 * color_height
    radius = 0.1 * color_width

    # Generate random position within the circular region
    theta = random.uniform(0, 2 * np.pi)
    r = random.uniform(0, radius)
    x_pos = int(center_x + r * np.cos(theta) - logo_width / 2)
    y_pos = int(center_y + r * np.sin(theta) - logo_height / 2)

    # Clamp positions to ensure logo is fully within bounds
    x_pos = max(0, min(color_width - logo_width, x_pos))
    y_pos = max(0, min(color_height - logo_height, y_pos))

    # Paste the logo onto the color image
    color.paste(logo, (x_pos, y_pos), logo)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    color.save(os.path.join(output_dir, "color.png"))

    # Get the alpha channel of the logo as a mask
    logo_alpha = np.array(logo.split()[3])
    logo_mask = (logo_alpha > 0).astype(np.uint8)

    # Create coordinate grids for the logo region
    yy, xx = np.meshgrid(np.arange(logo_height), np.arange(logo_width), indexing="ij")
    valid_coords = logo_mask[yy, xx].astype(bool)

    global_y = y_pos + yy[valid_coords]
    global_x = x_pos + xx[valid_coords]

    if normal_opengl is not None:
        normals = normal_opengl[global_y, global_x, :3]
        normals[:, 0] = int(0.5 * 255)
        normals[:, 1] = int(0.5 * 255)
        normals[:, 2] = int(255)
        normal_opengl[global_y, global_x, :3] = normals
        normal_opengl = Image.fromarray(np.clip(normal_opengl, 0, 255).astype(np.uint8))
        normal_opengl.save(os.path.join(output_dir, "normal.png"))

    if roughness is not None:
        roughness[global_y, global_x] = 255 * 0.52
        roughness = Image.fromarray(np.clip(roughness, 0, 255).astype(np.uint8))
        roughness.save(os.path.join(output_dir, "roughness.png"))

    print(f"Logo pasted and saved to {output_dir}")


def normalise_image(image):
    min_img, max_img = np.min(image), np.max(image)
    if max_img > 1:
        image = np.array(image) / 255  # [0, 1]
    image = 2 * image - 1  # [-1, 1]
    return image

def denormalise_image(image):
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    return image

def denormalise_image_torch(image):
    image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    return image

def channels_first(image):
    if len(image.shape) == 3:
        return np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 4:
        return np.transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError("Image must be either 3D or 4D tensor")

def channels_last(image):
    if len(image.shape) == 3:
        return np.transpose(image, (1, 2, 0))
    elif len(image.shape) == 4:
        return np.transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError("Image must be either 3D or 4D tensor")

def load_image_as_array(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32)
    return image

def post_process_image(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    decoded_image = channels_last(denormalise_image(image))
    return decoded_image


def torch_image_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Converts a torch image tensor (3 x H x W, range [0, 1], on CUDA) to a PIL image.

    Args:
        img_tensor (torch.Tensor): A 3 x H x W tensor on CUDA in range [0, 1].

    Returns:
        PIL.Image.Image: The corresponding PIL image.
    """
    # Move to CPU and clamp to [0, 1]
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    
    # Convert to [0, 255] and uint8
    img_tensor = (img_tensor * 255).to(torch.uint8)
    
    # Convert to H x W x C format for PIL
    img_numpy = img_tensor.permute(1, 2, 0).numpy()
    
    return Image.fromarray(img_numpy)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Converts a PIL image to a torch tensor in channel-first format (C, H, W)
    with values normalized to the range [0, 1].

    Args:
        pil_img (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Tensor of shape (C, H, W) and dtype float32.
    """
    transform = T.ToTensor()  # Converts to [0, 1] and (C, H, W)
    return transform(pil_img)

def numpy_to_pil(img_array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array to a PIL Image.
    
    Parameters:
        img_array (np.ndarray): Input array of shape (H, W, C) or (H, W)
            with dtype float32 (values in [0, 1]) or uint8 (values in [0, 255]).
            
    Returns:
        PIL.Image.Image: The corresponding PIL Image.
    """
    arr = img_array
    # If float array, assume values in [0, 1] and scale to [0, 255]
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype(np.uint8)
    # If not uint8 already, convert
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def pil_black_to_white_bg(pil_img: Image.Image) -> Image.Image:
    """
    Replace pure-black background in an RGB PIL image with white.

    Args:
        pil_img (PIL.Image.Image): any mode, will be converted to RGB.
    Returns:
        PIL.Image.Image: RGB image with black→white.
    """
    # Ensure RGB
    img = pil_img.convert("RGB")
    arr = np.array(img)

    # Build mask of all-black pixels
    black_mask = np.all(arr == [0, 0, 0], axis=-1)  # shape H×W, True where R=G=B=0

    # Paint those pixels white
    arr[black_mask] = [255, 255, 255]

    # Back to PIL
    return Image.fromarray(arr)