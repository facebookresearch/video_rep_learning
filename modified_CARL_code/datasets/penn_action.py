# Modified pre-processing routine to use no objects

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
# from datasets.new_data_augment import apply_data_augment, apply_ssl_data_augment

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

# add caching of a fixed number of variants per sample
class PennAction(torch.utils.data.Dataset):
    def __init__(self, cfg, split, dataset_name=None, mode="auto", sample_all=False, use_cache=True, nvariants=5):
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
        # OLD AUGMENTATION METHOD
        # Perform data-augmentation
        if self.cfg.SSL and self.mode=="train":
            self.data_preprocess = create_ssl_data_augment(cfg, augment=True)
        elif self.mode=="train":
            self.data_preprocess = create_data_augment(cfg, augment=True)
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)
        # NEW AUGMENTATION METHOD
        # if self.cfg.SSL:
        #     self.data_preprocess = apply_ssl_data_augment
        # else:
        #     self.data_preprocess = apply_data_augment
        self.augment = (self.mode=="train")

        if 'tcn' in cfg.TRAINING_ALGO:
            self.num_frames = self.num_frames // 2

        self.use_cache = use_cache
        self.nvariants = nvariants
        if self.use_cache:
            self.cache_dir = os.path.join(self.cfg.PATH_TO_DATASET, "cache")
            os.makedirs(self.cache_dir, exist_ok=True)
        if not self.augment:
            self.nvariants = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        t0 = time.time()

        # id = self.dataset[index]["id"]
        
        name = self.dataset[index]["name"]
        frame_label = self.dataset[index]["frame_label"]
        seq_len = self.dataset[index]["seq_len"]
        video_file = os.path.join(self.cfg.PATH_TO_DATASET, self.dataset[index]["video_file"])
        
        if self.use_cache:
            if self.cfg.SSL and not self.sample_all:
                sub_cache = 'ssl'
            else:
                sub_cache = 'not_ssl'
            cur_cache = os.path.join(self.cache_dir, sub_cache, self.dataset[index]["video_file"]+'_cache')
            os.makedirs(cur_cache, exist_ok=True)
            variants = os.listdir(cur_cache)
            nv = len(variants)
            if nv >= self.nvariants:
                i = random.randint(0, nv-1)
                cache_file = os.path.join(cur_cache, variants[i])
                with open(cache_file, 'rb') as f:
                    try:
                        ret_data = pickle.load(f)
                    except:
                        print('WARNING: corrupted cache file:')
                        print(cache_file)
                        exit()
                return ret_data
            else:
                out_cache = os.path.join(cur_cache, '%03i.pkl'%nv)

        video, _, info = read_video(video_file, pts_unit='sec')
        video = video.permute(0,3,1,2).float() / 255.0 # T H W C -> T C H W, [0,1] tensor

        # DEBUG - work towards caching the decoded video...
        # print(video)
        # print(video_file)
        # print(video.shape)

        # print(video_file + ' loading done')
        # print(time.time()-t0)
        # t0 = time.time()

        if self.cfg.SSL and not self.sample_all:
            t0 = time.time()
            names = [name, name]
            steps_0, chosen_step_0, video_mask0 = self.sample_frames(seq_len, self.num_frames)
            
            # print('video size:')
            # print(video.shape)
            # print('sample0: ' + str(time.time()-t0))
            # print('CPU: ' + str(psutil.Process().cpu_num()))
            # print('CPU Count: ' + str(psutil.cpu_count()))
            # print(psutil.cpu_percent(percpu=True))
            # print('-----')
            t0 = time.time()

            view_0 = self.data_preprocess(video[steps_0.long()])
            # view_0 = self.data_preprocess(video[steps_0.long()], self.cfg, self.augment)
            # view_0 = apply_ssl_data_augment(video[steps_0.long()], self.cfg, self.augment)

            # print('preproc0: ' + str(time.time()-t0))
            # print('CPU: ' + str(psutil.Process().cpu_num()))
            # print('-----')
            # t0 = time.time()

            label_0 = frame_label[chosen_step_0.long()]
            steps_1, chosen_step_1, video_mask1 = self.sample_frames(seq_len, self.num_frames, pre_steps=steps_0)

            # print('sample1')
            # print(time.time()-t0)
            # t0 = time.time()

            view_1 = self.data_preprocess(video[steps_1.long()])
            # view_1 = self.data_preprocess(video[steps_1.long()], self.cfg, self.augment)
            # view_1 = apply_ssl_data_augment(video[steps_1.long()], self.cfg, self.augment)

            # print('preproc1')
            # print(time.time()-t0)
            # t0 = time.time()

            label_1 = frame_label[chosen_step_1.long()]
            videos = torch.stack([view_0, view_1], dim=0)
            labels = torch.stack([label_0, label_1], dim=0)
            seq_lens = torch.tensor([seq_len, seq_len])
            chosen_steps = torch.stack([chosen_step_0, chosen_step_1], dim=0)
            video_mask = torch.stack([video_mask0, video_mask1], dim=0)

            # print('stacking')
            # print(time.time()-t0)
            # print('----------')
            # t0 = time.time()

            # print(video_file + ' sampling and preproc done')
            # print(time.time()-t0)
            # t0 = time.time()
            # print('----------')

            if self.use_cache:
                ret_data = (videos, labels, seq_lens, chosen_steps, video_mask, names)
                with open(out_cache, 'wb') as f:
                    pickle.dump(ret_data, f)
            return videos, labels, seq_lens, chosen_steps, video_mask, names

        elif not self.sample_all:
            steps, chosen_steps, video_mask = self.sample_frames(seq_len, self.num_frames)
        else:
            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            seq_len = len(steps)
            chosen_steps = steps.clone()
            video_mask = torch.ones(seq_len)

        # Select data based on steps
        video = video[steps.long()]
        
        video = self.data_preprocess(video)
        # video = self.data_preprocess(video, self.cfg, self.augment)

        if self.cfg.DATA.FRAME_LABELS:
            label = frame_label[chosen_steps.long()]

        if self.use_cache:
            ret_data = (video, label, torch.tensor(seq_len), chosen_steps, video_mask, name)
            with open(out_cache, 'wb') as f:
                pickle.dump(ret_data, f)
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