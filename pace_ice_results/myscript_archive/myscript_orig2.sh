#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH -J smolvlm-orig-inst-500M    # jobs name
#SBATCH -o ./slurm_outs/slurm_%j.out   # file to write logs, prints, etc

module load anaconda3/2023.03
conda activate /home/hice1/ryoshida7/scratch/py10_vlm

#python vlm_inference_orig.py --config config/config-inst-256M.yaml
python vlm_inference_orig.py --config config/config-inst-500M.yaml
#python vlm_inference_orig.py --config config/config-inst-2B.yaml
#python vlm_inference_orig.py --config config/config-base-256M.yaml
#python vlm_inference_orig.py --config config/config-base-500M.yaml
#python vlm_inference_orig.py --config config/config-base-2B.yaml