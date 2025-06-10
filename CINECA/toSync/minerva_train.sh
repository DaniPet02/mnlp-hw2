#!/bin/bash

#SBATCH --nodes=1                    # 1 node 
#SBATCH --partition=boost_usr_prod   # (3 node, 4-gpuXnode)
#SBATCH --cpus-per-task=32            # number of cpu per tasks
#SBATCH --ntasks-per-node=1          # 1 tasks per nod
#SBATCH --mem=32024                  # 32 GB ram
#SBATCH --time=2:30:00               # time limit: 1h
#SBATCH --error=./log/%x.err         # standard error file
#SBATCH --output=./log/%x.out        # standard output file
#SBATCH --account=try25_navigli      # project account
#SBATCH --job-name=minerva_training  # project value
#SBATCH --gres=gpu:2

module load profile/deeplrn
module load cuda/12.3
module load cudnn
module load python/3.11.6--gcc--8.5.0
source MNLP/bin/activate

export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wandb offline mode
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export FORCE_TORCHRUN=1
export CUDA_LAUNCH_BLOCKING=1

which python3
python3 -m site
python3 -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
python -c "import accelerate; print(accelerate.__version__)"

# Run fine-tuning
python3 train.py