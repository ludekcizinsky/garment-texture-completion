import os
import traceback

import pytorch_lightning as pl

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from helpers.pl_module import GarmentInpainterModule
from helpers.dataset import get_dataloaders
from helpers.data_utils import denormalise_image_torch
from helpers.metrics import compute_all_metrics

from tqdm import tqdm
import pandas as pd
import wandb

CKPT_ROOT = '/scratch/izar/cizinsky/garment-completion/checkpoints'


def load_checkpoint_and_cfg(run_name: str):
    checkpoint_path = f"{CKPT_ROOT}/{run_name}/last.ckpt"
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} does not exist"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["hyper_parameters"]
    return checkpoint, cfg

def load_model_and_data(cfg, checkpoint):

    trn_dataloader, val_dataloader = get_dataloaders(cfg)

    model = GarmentInpainterModule(cfg, trn_dataloader)
    model.setup()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to("cuda")

    return model, val_dataloader

def get_best_inference_setup_results(eval_cfg):

    checkpoint, cfg = load_checkpoint_and_cfg(eval_cfg.run_name)
    pl.seed_everything(cfg.seed)
    model, val_dataloader = load_model_and_data(cfg, checkpoint)
    val_batches = [next(iter(val_dataloader)) for _ in range(eval_cfg.max_n_batches)]

    rows = []
    for img_scale in eval_cfg.image_guidance_scale:
        for text_scale in eval_cfg.text_guidance_scale:
            sample_idx = 0
            for batch in tqdm(val_batches, desc=f"img_scale={img_scale}, text_scale={text_scale}"):
                pred_imgs = model.inference(batch["partial_diffuse_img"].to("cuda"), num_inference_steps=50, guidance_scale=text_scale, image_guidance_scale=img_scale)
                pred_imgs_tensors = torch.stack([pil_to_tensor(img) for img in pred_imgs]).to("cuda") / 255.0
                target_imgs = denormalise_image_torch(batch["full_diffuse_img"].to("cuda"))
                image_metrics = compute_all_metrics(pred_imgs_tensors, target_imgs)

                for i in range(len(pred_imgs)):
                    ith_ssim = image_metrics["ssim"][i]
                    ith_psnr = image_metrics["psnr"][i]
                    ith_lpips = image_metrics["lpips"][i]
                    rows.append({
                        "img_scale": img_scale,
                        "text_scale": text_scale,
                        "ssim": ith_ssim.item(),
                        "psnr": ith_psnr.item(),
                        "lpips": ith_lpips.item(),
                        "sample_idx": sample_idx,
                    })
                    sample_idx += 1
    df = pd.DataFrame(rows, columns=["sample_idx", "img_scale", "text_scale", "ssim", "psnr", "lpips"])

    os.makedirs(eval_cfg.output_dir, exist_ok=True)
    file_path = os.path.join(eval_cfg.output_dir, f"{eval_cfg.run_name}.csv")
    df.to_csv(file_path, index=False)

    run = wandb.init(entity=eval_cfg.entity, project=eval_cfg.project, id=eval_cfg.run_id, resume="must")
    wandb_table = wandb.Table(dataframe=df)
    run.log({"best_inference_setup": wandb_table})
    run.finish()


def run_post_train_evaluation(run_name, run_id, entity="ludekcizinsky", project="pbr-generation"):

    try:
        # Setup
        checkpoint, cfg = load_checkpoint_and_cfg(run_name)
        pl.seed_everything(cfg.seed)
        model, val_dataloader = load_model_and_data(cfg, checkpoint)
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False, callbacks=[])

        # Predict
        outputs = trainer.predict(model, val_dataloader)

        # Compute metrics
        all_ssim = torch.cat([output["ssim"] for output in outputs])
        all_psnr = torch.cat([output["psnr"] for output in outputs])
        all_lpips = torch.cat([output["lpips"] for output in outputs])

        mean_ssim = all_ssim.mean()
        mean_psnr = all_psnr.mean()
        mean_lpips = all_lpips.mean()

        # Log
        run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
        run.summary.update({
            "final_eval/ssim": mean_ssim,
            "final_eval/psnr": mean_psnr,
            "final_eval/lpips": mean_lpips
        })
        run.finish()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("-" * 50)
        wandb.finish()