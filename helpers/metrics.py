import torch
import torch.nn.functional as F
from torchmetrics.image import peak_signal_noise_ratio as psnr, structural_ssim as ssim
import lpips

# Initialize LPIPS model once
_LPIPS_MODEL = lpips.LPIPS(net='alex').eval()


def compute_ssim(preds: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute mean Structural Similarity Index Measure (SSIM) between preds and targets.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, data_range].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
        data_range (float): The data range of the inputs (max - min). Default is 1.0.
    
    Returns:
        float: SSIM score (averaged over the batch).
    """
    preds = preds.float()
    targets = targets.float()
    return ssim(preds, targets, data_range=data_range).item()


def compute_psnr(preds: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute mean Peak Signal-to-Noise Ratio (PSNR) between preds and targets.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, data_range].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
        data_range (float): The data range of the inputs (max - min). Default is 1.0.
    
    Returns:
        float: PSNR score in dB (averaged over the batch).
    """
    preds = preds.float()
    targets = targets.float()
    return psnr(preds, targets, data_range=data_range).item()


def compute_lpips(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean Learned Perceptual Image Patch Similarity (LPIPS) between preds and targets.
    
    Args:
        preds (torch.Tensor): Predicted images, shape [B, C, H, W], values in [0, 1].
        targets (torch.Tensor): Ground truth images, same shape and range as preds.
    
    Returns:
        float: LPIPS score (averaged over the batch).
    """
    # Convert from [0, 1] to [-1, 1] as expected by LPIPS
    preds_norm = preds.float() * 2 - 1
    targets_norm = targets.float() * 2 - 1
    with torch.no_grad():
        # LPIPS returns a [B, 1, 1, 1] tensor
        dist = _LPIPS_MODEL(preds_norm, targets_norm)
        return dist.mean().item()


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


# Example usage (for testing)
if __name__ == "__main__":
    B, C, H, W = 4, 3, 256, 256
    preds = torch.rand(B, C, H, W)
    targets = torch.rand(B, C, H, W)
    metrics = compute_all_metrics(preds, targets)
    print(metrics)
