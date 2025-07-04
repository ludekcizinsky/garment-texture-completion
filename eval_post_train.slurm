#!/bin/bash
#SBATCH --job-name=eval_post_train
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.out
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.err
#SBATCH --time=02:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion

# Part 1 (45 min each)
# Baseline
# python eval_post_train.py evaluation.run_name=fast-universe-159 evaluation.run_id=w5daifhx

# DDIM 
# python eval_post_train.py evaluation.run_name=twilight-sponge-160 evaluation.run_id=fstkreeo 
# python eval_post_train.py evaluation.run_name=sandy-capybara-119 evaluation.run_id=5ajtw75b
# python eval_post_train.py evaluation.run_name=eager-energy-111 evaluation.run_id=4rc12jo5

# LR 
# python eval_post_train.py evaluation.run_name=valiant-resonance-120 evaluation.run_id=6ei613pt
# python eval_post_train.py evaluation.run_name=honest-bird-113 evaluation.run_id=4dro7oe6

# EMA
# python eval_post_train.py evaluation.run_name=grateful-terrain-163 evaluation.run_id=2afus61b

# LR cos scheduler
# python eval_post_train.py evaluation.run_name=glowing-disco-164 evaluation.run_id=dg766dgh evaluation.use_cosine_scheduler=True

# From scratch
# python eval_post_train.py evaluation.run_name=stellar-feather-136 evaluation.run_id=y5appjkm evaluation.train_from_scratch=True

# Lora
# python eval_post_train.py evaluation.run_name=cosmic-cosmos-123 evaluation.run_id=b835hnfq evaluation.train_with_lora=True
# python eval_post_train.py evaluation.run_name=morning-microwave-118 evaluation.run_id=ha6wswub evaluation.train_with_lora=True

# From pretrained (instruct-pix2pix)
# python eval_post_train.py evaluation.run_name=null evaluation.run_id=null evaluation.use_pretrained_unet=True

# Inpainting
# - full unet
# python eval_post_train.py evaluation.run_name=honest-sound-182 evaluation.run_id=xwexoq8d 
# - lora
# python eval_post_train.py evaluation.run_name=woven-gorge-180 evaluation.run_id=uq7v05g2 

# Train ds scale
# - 5000
# python eval_post_train.py evaluation.run_name=lunar-night-196 evaluation.run_id=5cvjz9ox 
# - 10000
# python eval_post_train.py evaluation.run_name=azure-energy-197 evaluation.run_id=qf7pnzph 
# - 15000
# python eval_post_train.py evaluation.run_name=devoted-glade-198 evaluation.run_id=dhece0q4
# - 20000
# python eval_post_train.py evaluation.run_name=flowing-pyramid-199 evaluation.run_id=g8kycmqy

# Length of training
# valiant-resonance-120, before resume: final_eval/lpips=0.50461 final_eval/ssim=0.16205 final_eval/psnr=12.41427
# python eval_post_train.py evaluation.run_name=valiant-resonance-120 evaluation.run_id=6ei613pt 


# Eval of the best model on normal and roughness
python eval_post_train.py evaluation.run_name=valiant-resonance-120 evaluation.run_id=6ei613pt evaluation.full_pbr_inference=true