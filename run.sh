#!/usr/bin/env bash

#SBATCH -J "baby-detection"
#SBATCH -o logs/logs_%A.txt
#SBATCH --mem-per-cpu=1G
#SBATCH --nodes 2
#SBATCH --partition gpu
#SBATCH --gres=gpu:teslav100:2
#SBATCH --mail-user=<your_email_address>
#SBATCH --mail-type=ALL
#SBATCH --time=01:30:00

export PYTHONPATH=$PYTHONPATH:.

module load CUDA
python -u extract_world_view.py <recording_dir>