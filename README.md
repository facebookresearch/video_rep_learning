# Video Representation Learning

Self-Supervised Hierarchical Video Representation Learning

# CARL environment set up for AI4P

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
