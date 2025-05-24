#!/bin/bash
#SBATCH --job-name=eval_post_train
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.out
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.err
#SBATCH --time=07:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion

# DDIM (45 min each)
python eval_post_train.py evaluation.run_name=twilight-sponge-160 evaluation.run_id=fstkreeo 
python eval_post_train.py evaluation.run_name=sandy-capybara-119 evaluation.run_id=5ajtw75b
python eval_post_train.py evaluation.run_name=eager-energy-111 evaluation.run_id=4rc12jo5

# LR (45 min each)
python eval_post_train.py evaluation.run_name=valiant-resonance-120 evaluation.run_id=6ei613pt
python eval_post_train.py evaluation.run_name=honest-bird-113 evaluation.run_id=4dro7oe6

# EMA
python eval_post_train.py evaluation.run_name=grateful-terrain-163 evaluation.run_id=2afus61b
