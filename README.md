# Baby detection

Implement baby detection on video recorded from [Pupil glasses](https://pupil-labs.com/products/core/) ([documentations](https://docs.pupil-labs.com/core/)) using [Detectron2](https://github.com/facebookresearch/detectron2)

## 1. Setup the system

### 1.1 Conda environment 
[Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (a mini version of Anaconda)

```
conda create --name <env_name>
conda activate <env_name>
conda install -c anaconda cudatoolkit
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch` 
conda install pandas
pip install opencv-python
conda install ipykernel
conda install seaborn
```

### 1.2 Detectron2

#### 1.2.1 Linux

[Official guideline](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source)

1. Get GCC  
    **Note**: The following GCC guideline is just tested only on [narvi cluster](https://tuni-itc.github.io/wiki/Technical-Notes/tuni-narvi-cluster/#how-do-i-install-mysoftware)
    - Check GCC's version:  
      `gcc --version` or `g++ --version`
    - Check all available versions of GCC
      `module spider GCC`
    - Load any GCC version satisfying the [requirement](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#requirements).  
      `module load GCC/<gcc_version>`

2. Build detectron2  
    `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

#### 1.2.2 Window
Follow instruction from [this Detectron2's discussion](https://github.com/facebookresearch/detectron2/discussions/3308#discussion-3498102).  
TLDR:
   1. Create environment: `conda create -n <env_name>` followed by `conda activate <env_name>` 
   2. Cython: `pip install cython`
   3. Detectron2: Go to environment location (find from first command's output) and `git clone https://github.com/facebookresearch/detectron2.git`
   4. `python -m pip install -e detectron2`  

**Note**: If one encounters this error when trying to run `python baby_detection.py`
```
ImportError: DLL load failed while importing win32file: The specified procedure could not be found.
```
then also run `conda install pywin32`.

## 2. Software settings
Defined in [config.toml](config.toml):
- `model_config_path`: Pick from `all_model_config_paths` to use in the program
- `data_directory`: Pick from `all_data_directories` to use in the program
- `min_detection_score`: Minimum score of for model's predictions 
- `seed_number`: Seed number to generate pseudo-random numbers for reproductiveity

## 3. Run the programme
### 3.1 Linux (narvi cluster)
- Option 1: Run the following command in the terminal

```
srun \
--pty \
--job-name baby-detection \
--partition gpu \
--gres gpu:1 \
--mem-per-cpu 1G \
--ntasks 1 \
--cpus-per-task 10 \
--time 01:30:00 \
python baby_detection.py <recording_dir>
```

- Option 2: Create a bash file `run.sh` using the following template:

```
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
python baby_detection.py
```

and run the following command in the terminal:

```
sbatch run.sh
```

### 3.2 Window
`python baby_detection.py`
