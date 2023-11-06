# MV-Former

This is the code for MV-Former (Multi-entity Video Transformer) for self-supervised fine-grained video representation learning.

This code is forked from the Contrastive Action Representation Learning (CARL) codebase (https://github.com/minghchen/CARL_code) which is distributed under an MIT License.

# Environment Setup

```
# recommended conda pytorch setup for AWS:
conda create -y -n carl av pytorch=1.12.1 cudatoolkit=11.6 torchvision torchaudio \
--strict-channel-priority --override-channels \
-c https://aws-pytorch.s3.us-west-2.amazonaws.com \
-c pytorch \
-c nvidia \
-c conda-forge
conda activate carl

# repo requirements:
cd modified_CARL_code
pip install -r requirements.txt

# protobuf fix:
pip install protobuf==3.20.*

# PyAV fix:
pip install --force-reinstall av

# Update TIMM:
pip install timm==0.9.2

# Install Decord:
pip install decord
```

# Usage

First, download the necessary datasets by following the instructions in the CARL README.

To launch training, use the MV-Former config files provided in configs_mvf

```bash
cd modified_CARL_code
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs_mvf/penn_mvf.yml --logdir ~/penn_mvf
```