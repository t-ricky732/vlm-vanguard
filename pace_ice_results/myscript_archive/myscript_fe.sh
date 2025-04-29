#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:30:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH -J 256M-instruct-comp-ti-5epoch-21-2    # jobs name
#SBATCH -o ./slurm_outs/slurm_%j.out   # file to write logs, prints, etc

module load anaconda3/2023.03
conda activate /home/hice1/ryoshida7/scratch/py10_vlm

python vlm_train.py --config config_fe/config-256M-21.yaml
python vlm_inference.py --config config_fe/config-256M-21.yaml
