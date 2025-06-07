#!/bin/bash
# load base project modules

module load python/3.11.6--gcc--8.5.0
module load git

# creating a virtualenv, basically just a new directory (my_venv) containing all you need
ENV_NAME="MNLP"
TOKEN_HF="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"
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
    pip3 install -e ".[torch,metrics]" --no-build-isolation
    # hugging-face tools
    pip3 install -U "huggingface_hub[cli]"
    python3 -c "from huggingface_hub import login; login(token='$TOKEN_HF')"
else
    source MNLP/bin/activate
fi

# download LLMA-factory
if [ ! -d "LLaMA-Factory" ]; then 
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
# configure LLMA-factory
cp config.yaml LLaMA-Factory
cat dataset_config.json > LLaMA-Factory/data/dataset_info.json

# Enter in LLMA-factory folder
cd LLaMA-Factory

# download dataset via hugging-face hub
python3 -c "from datasets import load_dataset; load_dataset('GAIR/lima', cache_dir='LIMA')"




