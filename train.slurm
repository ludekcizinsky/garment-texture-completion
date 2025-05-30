#!/bin/bash
#SBATCH --job-name=pbr
#SBATCH --output=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.out
#SBATCH --error=/home/cizinsky/garment-texture-completion/outputs/slurm/%x.%j.err
#SBATCH --time=30:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1                

source /home/cizinsky/venvs/garment/bin/activate

cd /home/cizinsky/garment-texture-completion


# Part 1
# train further
python train.py optim.lr=1e-4 logger.run_id=6ei613pt 'logger.tags=[part1, lr]'


# Part 2
# python train.py model.is_inpainting=True data.batch_size=10 'logger.tags=[part2, baseline]'
# python train.py model.is_inpainting=True model.train_with_lora=True 'logger.tags=[part2, lora]'