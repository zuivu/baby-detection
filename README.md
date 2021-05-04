# Baby detection
Implement baby detection on video recorded from [Pupil glasses](https://pupil-labs.com/products/core/) ([documentations](https://docs.pupil-labs.com/core/)) using [Detectron2](https://github.com/facebookresearch/detectron2)

## 1. Setup the system
### 1.1 Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a virtual environment:
```
conda create --name <env_name>
conda activate <env_name>
conda install -c anaconda cudatoolkit
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pandas
pip install opencv-python
```    

### 1.2 Builiding detectron2 from source  
[Official guideline](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source)  
#### 1.2.1 Get GCC
**Note**: The following GCC guideline is used only for [narvi cluster](https://tuni-itc.github.io/wiki/Technical-Notes/tuni-narvi-cluster/#how-do-i-install-my-software)
- Check GCC's version:  
`gcc --version` or `g++ --version`
- Check all available versions of GCC  
`module spider GCC`
- Load any GCC version satisfying the [requirement](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#requirements).  
`module load GCC/<gcc_version>`

#### 1.2.2 Build detectron2
`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

## 2. Run detectron2
- Option 1:  
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
python extract_world_view.py <recording_dir>
```  
- Option 2: Before running the command, remember to change the email address  
```
sbatch run.sh
```
