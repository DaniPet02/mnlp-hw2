#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --partition=lrd_all_serial   # standard prod config
#SBATCH --qos=normal                 # normal resources
#SBATCH --cpus-per-task=1            # number of cpu per tasks
#SBATCH --ntasks-per-node=1          # 1 tasks per node                    
#SBATCH --mem=1024                   # 1GB
#SBATCH --time=0:10:00               # time limit: 10 min
#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=try25_navigli      # project account
#SBATCH --job-name=install           # project value


llamafactory-cli train ../toSync/config.yaml