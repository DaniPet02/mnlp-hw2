#!/bin/bash

#SBATCH --nodes=1                    # 1 node 
#SBATCH --partition=boost_usr_prod   # (3 node, 4-gpuXnode)
#SBATCH --qos=boost_qos_lprod        # GPU accelerated!
#SBATCH --cpus-per-task=8            # number of cpu per tasks
#SBATCH --ntasks-per-node=3          # 1 tasks per node                    
#SBATCH --mem=32024                   # 1GB
#SBATCH --time=1:00:00               # time limit: 10 min
#SBATCH --error=llma_factory/minerva/log/%x-%j.err            # standard error file
#SBATCH --output=llma_factory/minerva/log/%x-%j.out           # standard output file
#SBATCH --account=try25_navigli      # project account
#SBATCH --job-name=minerva_traning           # project value
#SBATCH --gresp:gpu:2

llamafactory-cli train config.yaml