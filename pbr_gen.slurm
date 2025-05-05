#!/bin/bash
#SBATCH --job-name=pbr_gen
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/pbr_gen_output.log
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/pbr_gen_error.log
#SBATCH --time=30:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           
#SBATCH --mem=16G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion

python pbr_gen.py
