#!/bin/bash
#SBATCH --job-name=pbr_train
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/pbr_train_baseline_output.log
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/pbr_train_baseline_error.log
#SBATCH --time=40:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion

python train.py 
