#!/bin/bash
#SBATCH --job-name=eval_inf_setup
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.out
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.err
#SBATCH --time=07:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion

python eval_inf_setup.py 