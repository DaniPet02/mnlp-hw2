#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH -p boost_usr_prod            # standard prod config
#SBATCH -q normal                    # normal resources
#SBATCH --cpus-per-task=1            # number of cpu per tasks
#SBATCH --ntasks-per-node=32         # 32 tasks per node                    # 
#SBATCH --mem 1024                   # 1GB
#SBATCH --time=0:10:00               # time limit: 10 min
#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=try25_navigli      # project account
#SBATCH --partition=<partition_name> # partition name
#SBATCH --qos=<qos_name>             # quality of service
#SBATCH --job-name="install"

# load project modules
module load python/3.10.8
module load git

# creating a virtualenv, basically just a new directory (my_venv) containing all you need
python3 -m venv MNLP

# activating the new virtualenv
source my_venv/bin/MNLP

# installing whatever you need (e.g matplotlib)
pip3 install scipy numpy nltk pandas pytest
pip3 install matplotlib wandb
pip3 install seaborn
pip3 install datasets

# create log directory