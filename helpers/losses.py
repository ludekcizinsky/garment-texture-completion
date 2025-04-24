import torch.nn.functional as F

def ddim_loss(alpha_t, latents, noisy_latents, model_pred):
    
    sqrt_alpha_t = alpha_t.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(-1, 1, 1, 1)

    # DDIM reverse step: estimate the noise-free latents at t=0
    ddim_pred = (
        noisy_latents  # Current latent scaled by next alpha  
        - sqrt_one_minus_alpha_t * model_pred  # Remove noise proportional to the next timestep
    ) / sqrt_alpha_t
    
    ddim_loss = F.mse_loss(ddim_pred, latents, reduction="mean")
    
    return ddim_loss