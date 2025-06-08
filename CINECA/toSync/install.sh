#!/bin/bash
# load base project modules

module load python/3.11.6--gcc--8.5.0
module load profile/deeplrn
module load  cudnn
module load cuda/12.3

module list
# creating a virtualenv, basically just a new directory (my_venv) containing all you need

# hugging-face tools
export WANDB_MODE=offline
ENV_NAME="MNLP"
TOKEN_HF="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"
TOKEN_WB="5f5206347e169e8bd0e1aba0680c8107aecac638"

# Delete old env
if [ ! -d "$ENV_NAME" ]; then
    echo "===[create MNLP]==="
    python3 -m venv MNLP
    
    # activating the new virtualenv
    source MNLP/bin/activate

    # upgrade pip
    python3 -m pip install --upgrade pip

    # installing whatever you need (e.g matplotlib)
    pip3 install numpy wandb datasets
    pip3 install --upgrade pip setuptools wheel
    pip3 install deepspeed
    pip3 install badam
    pip3 install -U "huggingface_hub[cli]"
    
else
    source MNLP/bin/activate
fi

# Setup External systems
python3 -c "from huggingface_hub import login; login(token='$TOKEN_HF')"
python3 -c "import wandb; wandb.login(key='$TOKEN_WB')"

# download LLMA-factory
if [ ! -d "LLaMA-Factory" ]; then 
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
# configure LLMA-factory
cp config.yaml LLaMA-Factory
cat dataset_config.json > LLaMA-Factory/data/dataset_info.json

# Enter in LLMA-factory folder
cd LLaMA-Factory
pip3 install -e ".[torch,metrics]" --no-build-isolation
mkdir -p minerva-7b
huggingface-cli download sapienzanlp/Minerva-7B-base-v1.0 \
  --local-dir minerva-7b \
  --local-dir-use-symlinks False


# download dataset via hugging-face hub
python3 -c "from datasets import load_dataset; load_dataset('llamafactory/lima', cache_dir='LIMA')"




