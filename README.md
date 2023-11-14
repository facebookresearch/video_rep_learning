# MV-Former

This is the code for MV-Former (Multi-entity Video Transformer) for self-supervised fine-grained video representation learning.

This code is forked from the Contrastive Action Representation Learning (CARL) codebase (https://github.com/minghchen/CARL_code).

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
cd CARL_MVF
pip install -r requirements.txt
pip install protobuf==3.20.*
pip install --force-reinstall av
pip install timm==0.9.2
pip install decord
```

# Usage

First, download the necessary datasets by following the instructions in CARL_MVF/README.md.

To launch training, use the MV-Former config files provided in CARL_MVF/configs_mvf/

```bash
cd CARL_MVF
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs_mvf/penn_mvf.yml --logdir ~/penn_mvf
```

# License

The majority of MV-Former is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/minghchen/CARL_code is licensed under the MIT license.