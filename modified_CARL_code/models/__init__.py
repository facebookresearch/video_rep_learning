import torch
from models.resnet_c2d import BaseModel, Classifier
from models.transformer import TransformerModel
import utils.logging as logging

logger = logging.get_logger(__name__)

def build_model(cfg, local_rank=None):
    if cfg.MODEL.EMBEDDER_TYPE == "transformer":
        model = TransformerModel(cfg, local_rank)
    else:
        model = BaseModel(cfg)
    return model

import os

def save_checkpoint(cfg, model, optimizer, epoch):
    path = os.path.join(cfg.LOGDIR, "checkpoints")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, "checkpoint_epoch_{:05d}.pth".format(epoch))
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Saving epoch {epoch} checkpoint at {ckpt_path}")

# ORIGINAL LOADER
# def load_checkpoint(cfg, model, optimizer):
#     path = os.path.join(cfg.LOGDIR, "checkpoints")
#     if os.path.exists(path):
#         names = [f for f in os.listdir(path) if "checkpoint" in f]
#         if len(names) > 0:
#             name = sorted(names)[-1]
#             ckpt_path = os.path.join(path, name)
#             logger.info(f"Loading checkpoint at {ckpt_path}")
#             checkpoint = torch.load(ckpt_path)
#             model.module.load_state_dict(checkpoint["model_state"])
#             optimizer.load_state_dict(checkpoint["optimizer_state"])
#             # cfg.update(checkpoint["cfg"])
#             # return checkpoint["epoch"]
#             # bug fix: checkpoint number should be incremented by 1 for next epoch
#             return checkpoint["epoch"] + 1
#     return 0

# NEW: for k400 pretraining, added support to load a checkpoint from another dir also
# which must be specified by cfg.MODEL.PRETRAINED_CHECKPOINT. Will only load from said
# checkpoint if the model is starting for the first time. If an existing checkpoint is
# found in the current logdir, it will be loaded instead
def load_checkpoint(cfg, model, optimizer):
    path = os.path.join(cfg.LOGDIR, "checkpoints")
    if os.path.exists(path):
        names = [f for f in os.listdir(path) if "checkpoint" in f]
        if len(names) > 0:
            name = sorted(names)[-1]
            ckpt_path = os.path.join(path, name)
            logger.info(f"Loading checkpoint at {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model.module.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # cfg.update(checkpoint["cfg"])
            # return checkpoint["epoch"]
            # bug fix: checkpoint number should be incremented by 1 for next epoch
            return checkpoint["epoch"] + 1
    # NEW: load pre-trained checkpoint
    if "PRETRAINED_CHECKPOINT" in cfg.MODEL and cfg.MODEL.PRETRAINED_CHECKPOINT is not None:
        ckpt_path = cfg.MODEL.PRETRAINED_CHECKPOINT
        if not os.path.exists(ckpt_path):
            print('ERROR: invalid path specified for cfg.MODEL.PRETRAINED_CHECKPOINT')
            print('could not find checkpoint at: ' + ckpt_path)
            exit(-1)
        logger.info(f"Loading pretrained checkpoint at {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        model.module.load_state_dict(checkpoint["model_state"])
        # do NOT load optimizer state dict for fine-tuning
    return 0