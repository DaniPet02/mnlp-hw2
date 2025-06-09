#!/bin/bash

#SBATCH --nodes=1                    # 1 node 
#SBATCH --partition=boost_usr_prod   # (3 node, 4-gpuXnode)
#SBATCH --cpus-per-task=16            # number of cpu per tasks
#SBATCH --ntasks-per-node=2          # 1 tasks per nod
#SBATCH --mem=32024                   # 1GB
#SBATCH --time=1:00:00               # time limit: 10h
#SBATCH --error=%x.err            # standard error file
#SBATCH --output=%x.out           # standard output file
#SBATCH --account=try25_navigli      # project account
#SBATCH --job-name=minerva_training           # project value
#SBATCH --gres=gpu:2

# Wandb offline mode
export WANDB_MODE=offline

# Run fine-tunning
llamafactory-cli train config.yaml