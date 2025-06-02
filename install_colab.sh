#!/bin/bash
# Ensure if all base software result installed

# Transformers additional reqired packages
pip3 install  bitsandbytes
pip3 install sentencepiece
# Dependecies for Opus Models
pip3 install sacremoses

# Update old version
pip3 install --upgrade accelerate
pip3 install -U datasets