# coding=utf-8
"""Visualize the attention of smart tokens. Based on evaluate.py"""

import os
import math
import torch
import pprint
import numpy as np
from tqdm import tqdm
import utils.logging as logging
from torch.utils.tensorboard import SummaryWriter

from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer
from datasets import construct_dataloader
from evaluation import get_tasks
from visualize_alignment import create_video, create_single_video, create_multiple_video
from visualize_retrieval import create_retrieval_video

from PIL import Image

# TODO - clean up import block

# logger = logging.get_logger(__name__)




# TODO - move to shared location
# generic feature extactor wrapper for intermedia layers modified
# from: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
# NEW: if return_output=True, returns the original model output along with
# the specified layer_id
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layer_id, return_output=True):
        super().__init__()
        self.model = model
        self.layer = layer_id
        self.return_output = return_output
        self._feature = torch.empty(0)
        layer = dict([*self.model.named_modules()])[layer_id]
        layer.register_forward_hook(self.save_outputs_hook())

    def save_outputs_hook(self):
        def fn(_, __, output):
            self._feature = output
        return fn

    # TODO - non-generic
    def forward(self, x, ns):
        out = self.model(x, ns)
        if self.return_output:
            return self._feature, out
        else:
            return self._feature



# TODO - remove extra args
def run_vis(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                    iterator_tasks, embedding_tasks, cur_epoch, summary_writer):
    attn_extractor = FeatureExtractor(model, 'module.embed.pooling.cross_att.attn_holder')

    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS
    if num_contexts != 1:
        print('num_contexts != 1 not supported')
        exit(-1)

    # get config name
    temp = cfg.args.cfg_file
    temp = os.path.splitext(temp)[0]
    temp = temp.split('/')[-1]
    config_name = temp

    # visualize for each dataset
    nds = len(val_emb_loader)
    suff = None
    for data_i in range(len(val_emb_loader)):
        data_loader = val_emb_loader[data_i]
        if nds > 1:
            suff = 'dataset%02i'%data_i
            print('dataset %i'%data_i)
        with torch.no_grad():
            for video, frame_label, seq_len, chosen_steps, video_masks, names in data_loader:
                assert video.size(0) == 1 # batch_size==1
                assert video.size(1) == frame_label.size(1) == int(seq_len.item())
                attns = []
                seq_len = seq_len.item()
                num_batches = int(math.ceil(float(seq_len)/max_frames_per_batch))
                frames_per_batch = int(math.ceil(float(seq_len)/num_batches))
                for i in range(num_batches):
                    curr_idx = i * frames_per_batch
                    num_steps = min(seq_len - curr_idx, frames_per_batch)
                    steps = torch.arange(curr_idx, curr_idx+num_steps)
                    steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
                    curr_data = video[:, steps]
                    # print(i, num_steps, seq_len, curr_data.shape)
                    if cfg.USE_AMP:
                        with torch.cuda.amp.autocast():
                            attn, _ = attn_extractor(curr_data, num_steps)
                    else:
                        attn, _ = attn_extractor(curr_data, num_steps)
                    attns.append(attn.cpu())
                attns = torch.cat(attns, dim=0)
                valid = (frame_label[0]>=0)
                attns = attns[valid].numpy()
                gen_vis(attns, config_name, suff=suff, video_frames=curr_data)                
                break
                


def gen_vis(attns, model_name='temp', out_dir='smart_token_vis', upscale=10, suff=None, video_frames=None):
    if video_frames is not None:
        video_frames = video_frames.cpu().numpy()[0,...]
        vf_min = np.min(video_frames)
        vf_max = np.max(video_frames)
        video_frames -= vf_min
        video_frames /= (vf_max - vf_min)
        video_frames = (video_frames*255).astype(np.uint8)
    n_frames, n_toks, n_pos = attns.shape
    hw = int(math.sqrt(n_pos))
    if (hw * hw) != n_pos:
        print('WARNING: can only visualize with a square token array')
        exit(-1) 
    cur_out_dir = os.path.join(out_dir, model_name)
    os.makedirs(cur_out_dir, exist_ok=True)
    ims = []
    vbuffer = None
    for nf in range(n_frames):
        ims_f = []
        if video_frames is not None:
            ims_f.append(video_frames[nf,...])
            w = video_frames.shape[2]
            h = video_frames.shape[3]
            if vbuffer is None:
                vbuffer = np.ones([3,h,10], dtype=np.uint8)*255
            ims_f.append(vbuffer)
        for nt in range(n_toks):
            v = attns[nf, nt, :]
            v = np.reshape(v, [hw, hw])
            v_min = np.min(v)
            v_max = np.max(v)
            v -= v_min
            v /= (v_max - v_min)
            im = Image.fromarray(np.uint8(v*255))
            # upscaling
            w, h = im.size
            if video_frames is not None:
                w = video_frames.shape[2]
                h = video_frames.shape[3]
            else:
                w *= upscale
                h *= upscale
            im = im.resize((w,h), resample=Image.NEAREST)
            im = np.array(im)
            im = np.stack([im,im,im], axis=0)
            ims_f.append(np.array(im))
            if vbuffer is None:
                vbuffer = np.ones([3,h,10], dtype=np.uint8)*255
            ims_f.append(vbuffer)
        ims_f.pop(-1)
        im_f = np.concatenate(ims_f, axis=2)
        im_f = np.moveaxis(im_f, 0, -1)
        im_f = Image.fromarray(im_f)
        ims.append(im_f)
    # save gif
    if suff is None:
        gif_name = os.path.join(cur_out_dir, '%s.gif'%model_name)
    else:
        gif_name = os.path.join(cur_out_dir, '%s_%s.gif'%(model_name, suff))
    # duration = second per frame
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], duration=1, loop=0)



def visualize():
    """Evaluate embeddings."""
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    # Setup summary writer.
    # summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'eval_logs'))
    summary_writer = None # TODO clean-up

    # Print config.
    # logger.info("Train with config:")
    # logger.info(pprint.pformat(cfg))
    # print("Train with config:")
    # print(pprint.pformat(cfg))

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=False)
    optimizer = construct_optimizer(model, cfg)
    start_epoch = load_checkpoint(cfg, model, optimizer)

    # Setup Dataset Iterators from train and val datasets.
    train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    val_loader, val_emb_loader = construct_dataloader(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    # TODO - clean up extra args
    run_vis(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                        iterator_tasks, embedding_tasks, start_epoch, summary_writer)

if __name__ == '__main__':
    visualize()
