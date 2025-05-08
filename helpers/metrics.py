import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr, learned_perceptual_image_patch_similarity as lpips

def compute_ssim(preds: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Structural Similarity Index Measure (SSIM) between preds and targets for each sample.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, data_range].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
        data_range (float): The data range of the inputs (max - min). Default is 1.0.
    
    Returns:
        torch.Tensor: SSIM scores for each sample in the batch, shape [B].
    """
    preds = preds.float()
    targets = targets.float()
    result = ssim(preds, targets, data_range=data_range, reduction=None)
    return result


def compute_psnr(preds: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between preds and targets for each sample.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, data_range].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
        data_range (float): The data range of the inputs (max - min). Default is 1.0.
    
    Returns:
        torch.Tensor: PSNR scores in dB for each sample in the batch, shape [B].
    """
    preds = preds.float()
    targets = targets.float()
    result = psnr(preds, targets, data_range=data_range, reduction=None, dim=[1, 2, 3])
    return result


def compute_lpips(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) between preds and targets for each sample.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, 1].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
    
    Returns:
        torch.Tensor: LPIPS scores for each sample in the batch, shape [B].
    """
    preds = preds.float()
    targets = targets.float()
    result = lpips(preds, targets, reduction="sum", normalize=True, net_type='alex')
    return result


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> dict:
    """
    Compute SSIM, PSNR, and LPIPS metrics between preds and targets.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W].
        targets (torch.Tensor): Ground truth images, same shape as preds.
        data_range (float): Data range of images for SSIM/PSNR. Default is 1.0.
    
    Returns:
        dict: {'ssim': float, 'psnr': float, 'lpips': float}
    """
    return {
        'ssim': compute_ssim(preds, targets, data_range),
        'psnr': compute_psnr(preds, targets, data_range),
        'lpips': compute_lpips(preds, targets),
    }
