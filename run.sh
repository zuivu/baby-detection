#!/usr/bin/env bash

#SBATCH -J "baby-detection"
#SBATCH -o logs/logs_%A.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCh --cpus-per-task=10
#SBATCH --mail-user=<your_email_address>
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00

export PYTHONPATH=$PYTHONPATH:.

module load CUDA
source activate <env_name>
python extract_world_view.py <recording_dir>
