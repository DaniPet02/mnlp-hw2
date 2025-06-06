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

# load base project modules

module load python/3.11.6--gcc--8.5.0
module load git

# creating a virtualenv, basically just a new directory (my_venv) containing all you need
ENV_NAME="MNLP"

# Delete old env
if [ -d "$ENV_NAME" ]; then
    echo "remove old env $ENV"
    rm -rf "$ENV_NAME"
fi

python3 -m venv MNLP

# activating the new virtualenv
source MNLP/bin/activate

# installing whatever you need (e.g matplotlib)
python3 -m pip install --upgrade pip
pip3 install scipy numpy nltk pandas pytest
pip3 install matplotlib wandb
pip3 install seaborn
pip3 install datasets

# create log directory
