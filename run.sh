#!/usr/bin/env bash

#SBATCH -J "Detectron2"
#SBATCH -o logs/logs_%A.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 2
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:2
#SBATCH --mail-user=[YOUR-EMAIL-ADDRESS]
#SBATCH --mail-type=ALL
#SBATCH -t 2-23:59:00

export PYTHONPATH=$PYTHONPATH:.

module load CUDA
python -u extract_world_view.py 