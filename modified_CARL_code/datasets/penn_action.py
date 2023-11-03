# coding=utf-8
import os
import math
import pickle
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import time
from torchvision.io import read_video
import random

import psutil

import utils.logging as logging
from datasets.data_augment import create_data_augment, create_ssl_data_augment

# from utils.dali_loader import dali_load

# EXPERIMENTAL - TODO - LOCAL SCRATCH CACHE
# USE_SCRATCH_CACHE = False

logger = logging.get_logger(__name__)

PENN_ACTION_LIST = [
    'baseball_pitch',
    'baseball_swing',
    'bench_press',
    'bowl',
    'clean_and_jerk',
    'golf_swing',
    'jumping_jacks',
    'pushup',
    'pullup',
    'situp',
    'squat',
    'tennis_forehand',
    'tennis_serve'
]


class PennAction(torch.utils.data.Dataset):
    def __init__(self, cfg, split, dataset_name=None, mode="auto", sample_all=False):
        assert split in ["train", "val", "test"]
        self.cfg = cfg
        self.split = split
        if mode == "auto":
            self.mode = "train" if self.split=="train" else "eval"
        else:
            self.mode = mode
        self.sample_all = sample_all
        self.num_contexts = cfg.DATA.NUM_CONTEXTS

        with open(os.path.join(cfg.PATH_TO_DATASET, split+'.pkl'), 'rb') as f:
            self.dataset, self.action_to_indices = pickle.load(f)

        if dataset_name is not None:
            indices = self.action_to_indices[PENN_ACTION_LIST.index(dataset_name)]
            self.dataset = [self.dataset[index] for index in indices]
            logger.info(f"{len(self.dataset)} {self.split} samples of {dataset_name} dataset have been read.")
        else:
            logger.info(f"{len(self.dataset)} {self.split} samples of Penn Action dataset have been read.")
        if not self.sample_all:
            seq_lens = [data['seq_len'] for data in self.dataset]
            hist, bins = np.histogram(seq_lens, bins='auto')
            print(list(bins.astype(np.int)))
            print(list(hist))

        self.num_frames = cfg.TRAIN.NUM_FRAMES
        
        # Perform data-augmentation
        # NEW - for training, preproc has been moved to run GPU-side for efficiency
        if self.cfg.SSL and self.mode=="train":
            # self.data_preprocess = create_ssl_data_augment(cfg, augment=True)
            self.data_preprocess = None
        elif self.mode=="train":
            # self.data_preprocess = create_data_augment(cfg, augment=True)
            self.data_preprocess = None
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)

        self.augment = (self.mode=="train")

        if 'tcn' in cfg.TRAINING_ALGO:
            self.num_frames = self.num_frames // 2

        # NEW - EXPERIMENTAL - SCRATCH CACHE
        # if USE_SCRATCH_CACHE:
        #     print('SCRATCH CACHE ENABLED')
        # TODO - REMOVE



    def __len__(self):
        return len(self.dataset)



    # NEW modified to return videos without pre-processing, so preprocessing can be handled on GPU-side
    def __getitem__(self, index):
        t0 = time.time()

        # id = self.dataset[index]["id"]
        
        name = self.dataset[index]["name"]
        frame_label = self.dataset[index]["frame_label"]
        seq_len = self.dataset[index]["seq_len"]
        video_file = os.path.join(self.cfg.PATH_TO_DATASET, self.dataset[index]["video_file"])

        # # NEW CACHE SYSTEM FOR DECODED VIDEOS ON LOCAL DRIVE
        # # TODO - REMOVE
        # cache_dir = '/scratch/mwalmer/temp/penn_action/' # TODO - add setting
        # cache_file = os.path.join(cache_dir, self.dataset[index]["video_file"]+'_cache.pkl')
        # if USE_SCRATCH_CACHE and os.path.isfile(cache_file):
        #     with open(cache_file, 'rb') as f:
        #         video = pickle.load(f)
        # else:
        #     video, _, info = read_video(video_file, pts_unit='sec')
        #     video = video.permute(0,3,1,2).float() / 255.0 # T H W C -> T C H W, [0,1] tensor
        #     # NEW
        #     if USE_SCRATCH_CACHE:
        #         cur_cache, _ = os.path.split(cache_file)
        #         os.makedirs(cur_cache, exist_ok=True)
        #         with open(cache_file, 'wb') as f:
        #             pickle.dump(video, f)

        if self.cfg.SSL and not self.sample_all:
            
            # NEW - fast data loading with NVIDIA DALI
            steps_0, chosen_step_0, video_mask0 = self.sample_frames(seq_len, self.num_frames)
            steps_1, chosen_step_1, video_mask1 = self.sample_frames(seq_len, self.num_frames, pre_steps=steps_0)
            s_start = min(int(steps_0[0]), int(steps_1[0]))
            s_stop = max(int(steps_0[-1]), int(steps_1[-1]))
            # video = dali_load(video_file, s_start, s_stop+1)
            # steps_0 -= s_start
            # steps_1 -= s_start
            video, _, info = read_video(video_file, pts_unit='sec')
            view_0 = video[steps_0]
            view_1 = video[steps_1]
            view_0 = view_0.permute(0,3,1,2).float() / 255.0 # T C H W, [0,1] tensor
            view_1 = view_1.permute(0,3,1,2).float() / 255.0 # T C H W, [0,1] tensor
            videos = (view_0, view_1)

            # NOTE - moved pre-proc to run GPU-side for efficiency
            names = [name, name]
            # steps_0, chosen_step_0, video_mask0 = self.sample_frames(seq_len, self.num_frames)
            # view_0 = video[steps_0.long()]
            label_0 = frame_label[chosen_step_0.long()]
            # steps_1, chosen_step_1, video_mask1 = self.sample_frames(seq_len, self.num_frames, pre_steps=steps_0)
            # view_1 = self.data_preprocess(video[steps_1.long()])
            # view_1 = video[steps_1.long()]
            label_1 = frame_label[chosen_step_1.long()]
            # videos = torch.stack([view_0, view_1], dim=0)
            # videos = (view_0, view_1)
            labels = torch.stack([label_0, label_1], dim=0)
            seq_lens = torch.tensor([seq_len, seq_len])
            chosen_steps = torch.stack([chosen_step_0, chosen_step_1], dim=0)
            video_mask = torch.stack([video_mask0, video_mask1], dim=0)

            return videos, labels, seq_lens, chosen_steps, video_mask, names

        elif not self.sample_all:
            steps, chosen_steps, video_mask = self.sample_frames(seq_len, self.num_frames)
        else:
            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            seq_len = len(steps)
            chosen_steps = steps.clone()
            video_mask = torch.ones(seq_len)

        # Select data based on steps
        # print('DEBUG - THIS HAPPENDED')
        # print(video.shape)

        # load old way:
        # video, _, info = read_video(video_file, pts_unit='sec')
        # video = video.permute(0,3,1,2).float() / 255.0 
        # print(video.shape)

        # load new way:
        # video = dali_load(video_file, steps[0], steps[-1]+1)
        video, _, info = read_video(video_file, pts_unit='sec')
        video = video.permute(0,3,1,2).float() / 255.0
        # print(video.shape)
        # print('-')

        video = video[steps.long()]
        video = self.data_preprocess(video)

        if self.cfg.DATA.FRAME_LABELS:
            label = frame_label[chosen_steps.long()]

        return video, label, torch.tensor(seq_len), chosen_steps, video_mask, name



    def sample_frames(self, seq_len, num_frames, pre_steps=None):
        # When dealing with very long videos we can choose to sub-sample to fit
        # data in memory. But be aware this also evaluates over a subset of frames.
        # Subsampling the validation set videos when reporting performance is not
        # recommended.
        sampling_strategy = self.cfg.DATA.SAMPLING_STRATEGY
        pre_offset = min(pre_steps) if pre_steps is not None else None
        
        if sampling_strategy == 'offset_uniform':
            # Sample a random offset less than a provided max offset. Among all frames
            # higher than the chosen offset, randomly sample num_frames
            if seq_len >= num_frames:
                steps = torch.randperm(seq_len) # Returns a random permutation of integers from 0 to n - 1.
                steps = torch.sort(steps[:num_frames])[0]
            else:
                steps = torch.arange(0, num_frames)
        elif sampling_strategy == 'time_augment':
            num_valid = min(seq_len, num_frames)
            expand_ratio = np.random.uniform(low=1.0, high=self.cfg.DATA.SAMPLING_REGION) if self.cfg.DATA.SAMPLING_REGION>1 else 1.0

            block_size = math.ceil(expand_ratio*seq_len)
            if pre_steps is not None and self.cfg.DATA.CONSISTENT_OFFSET != 0:
                shift = int((1-self.cfg.DATA.CONSISTENT_OFFSET)*num_valid)
                offset = np.random.randint(low=max(0, min(seq_len-block_size, pre_offset-shift)), high=max(1, min(seq_len-block_size+1, pre_offset+shift+1)))
            else:
                offset = np.random.randint(low=0, high=max(seq_len-block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames-num_valid), "constant", seq_len)
        else:
            raise ValueError('Sampling strategy %s is unknown. Supported values are '
                            'stride, offset_uniform .' % sampling_strategy)

        if 'tcn' in self.cfg.TRAINING_ALGO:
            pos_window = self.cfg.TCN.POSITIVE_WINDOW
            pos_steps = steps + torch.randint(-pos_window, 0, steps.size())
            steps = torch.stack([steps, pos_steps], dim=0)
            steps = steps.transpose(0, 1).contiguous().view(-1)
            num_frames = num_frames*2

        video_mask = torch.ones(num_frames)
        video_mask[steps<0] = 0
        video_mask[steps>=seq_len] = 0
        # Store chosen indices.
        chosen_steps = torch.clamp(steps.clone(), 0, seq_len - 1)
        if self.num_contexts == 1:
            steps = chosen_steps
        else:
            # Get multiple context steps depending on config at selected steps.
            context_stride = self.cfg.DATA.CONTEXT_STRIDE
            steps = steps.view(-1,1) + context_stride*torch.arange(-(self.num_contexts-1), 1).view(1,-1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)

        return steps, chosen_steps, video_mask


class ActionBatchSampler(torch.utils.data.Sampler):
    def __init__(self, cfg, dataset, batch_size, shuffle=True, seed=0):
        self.dist = True if cfg.NUM_GPUS > 1 else False
        self.dataset = dataset
        self.action_to_indices = dataset.action_to_indices
        self.batch_size = batch_size
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        batch = []
        action = -1
        for i in iter(range(self.num_samples)):
            if action == -1:
                action = torch.randint(high=len(self.action_to_indices), size=(1,), dtype=torch.int64)
                indices = self.action_to_indices[action.item()]
                indices_shuffle = torch.randperm(len(indices)).tolist()
            idx = indices[indices_shuffle[len(batch)]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                action = -1
        
    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch