# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.resnet_c2d import *
from models.utils import *
from models.mvformer import *


class TransformerModel(nn.Module):
    def __init__(self, cfg, local_rank=None):
        super().__init__()
        self.cfg = cfg

        # NEW - control fusion type
        if 'FUSION_TYPE' not in cfg.MODEL.EMBEDDER_MODEL:
            self.fusion_type = 'late'
        else:
            self.fusion_type = cfg.MODEL.EMBEDDER_MODEL.FUSION_TYPE
        
        # NEW - identify backbone type
        self.backbone_type = None

        # NEW - optional CLS_RES connection
        self.use_cls_res = False
        if 'CLS_RES' in self.cfg.MODEL and self.cfg.MODEL.CLS_RES:
            self.use_cls_res = True
            if self.fusion_type == 'late':
                print('ERROR: CLS_RES cannot be used with late fusion')
                exit(-1)

        # Backbone
        if 'TIMM-' in cfg.MODEL.BASE_MODEL.NETWORK: # NEW - TIMM Model Backbones
            self.backbone_type = 'timm'
            model_name = cfg.MODEL.BASE_MODEL.NETWORK[5:]
            
            if model_name in ['vit_small_patch16_224.dino', 'vit_small_patch8_224.dino', 'vit_small_patch14_dinov2.lvd142m']:
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 384
                blk_count = 12
            elif model_name in ['vit_base_patch16_224.dino', 'vit_base_patch8_224.dino', 'vit_base_patch14_dinov2.lvd142m']:
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 768
                blk_count = 12
            elif model_name == 'vit_large_patch14_dinov2.lvd142m':
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 1024
                blk_count = 24
            elif model_name == 'vit_giant_patch14_dinov2.lvd142m':
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 1536
                blk_count = 40
            else:
                print('ERROR: unknown/unsupported TIMM model:')
                print(model_name)
                exit()
            model = timm.create_model(model_name, pretrained=True)
            print('loaded TIMM backbone: ' + model_name)
                
            # NEW OPTIONAL RESIDUAL CONNECTION FOR CLS OUTPUT
            if self.use_cls_res:
                self.cls_res_res = nn.Linear(cfg.MODEL.BASE_MODEL.OUT_CHANNEL, cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)

            # NEW OPTIONAL: when using original model (self.fusion_type='late') can use either CLS features or spatial features
            self.late_type = 'cls'
            if 'LATE_TYPE' in cfg.MODEL.EMBEDDER_MODEL:
                self.late_type = cfg.MODEL.EMBEDDER_MODEL.LATE_TYPE
                assert self.late_type in ['cls', 'spatial']

            # Select layers to extract and stack if selecting multiple layers.
            # Setting is specified as either as a single block number or a comma-separated list of block numbers (i.e. "9,10,11")
            if self.fusion_type != 'late' or self.late_type == 'spatial':
                if 'SMART_FEATS' not in cfg.MODEL.EMBEDDER_MODEL:
                    print('MODEL.EMBEDDER_MODEL.SMART_FEATS not specified') 
                    print('extracting block 11 output spatial token features')
                    extract_ids = ['blocks.11']
                else:
                    extract_ids = []
                    temp = str(cfg.MODEL.EMBEDDER_MODEL.SMART_FEATS)
                    if ',' in temp:
                        temp = temp.split(',')
                    else:
                        temp = [temp]
                    for t_id in temp:
                        extract_ids.append('blocks.%s'%t_id)
                    print('extracting features for the following layers:')
                    print(extract_ids)
                    cfg.MODEL.BASE_MODEL.OUT_CHANNEL *= len(extract_ids)

            # fully frozen backbone
            if cfg.MODEL.BASE_MODEL.LAYER < 0 or cfg.MODEL.BASE_MODEL.LAYER >= blk_count:
                # fully frozen
                print('backbone fully frozen')
                if self.fusion_type != 'late' or self.late_type == 'spatial':
                    model = FeatureExtractor(model, extract_ids)
                self.backbone = model
                self.res_finetune = nn.Identity()
            # partially frozen backbone with splitting
            else:
                self.backbone = ViTFrontEnd(model, cfg.MODEL.BASE_MODEL.LAYER)
                self.res_finetune = ViTBackEnd(model, cfg.MODEL.BASE_MODEL.LAYER, local_rank)
                if self.fusion_type != 'late':
                    # update extract block id's to index into ViTBackEnd
                    new_extract_ids = []
                    for e_id in extract_ids:
                        b_idx = int(e_id.split('.')[-1])
                        b_idx -= cfg.MODEL.BASE_MODEL.LAYER
                        if b_idx < 0:
                            print('ERROR: cannot request extract of %s as it is not in ViTBackEnd wrapper'%e_id)
                            exit(-1)
                        e_id_new = 'blocks.%i'%b_idx
                        new_extract_ids.append(e_id_new)
                    # extract specified layer(s)
                    self.res_finetune = FeatureExtractor(self.res_finetune, new_extract_ids)

        else: # RESNET Backbones (original CARL)
            self.backbone_type = 'resnet'
            res50_model = models.resnet50(pretrained=True)
            load_pretrained_resnet50(cfg, res50_model)
            if cfg.MODEL.BASE_MODEL.LAYER == 3:
                self.backbone = nn.Sequential(*list(res50_model.children())[:-3]) # output of layer3: 1024x14x14
                self.res_finetune = list(res50_model.children())[-3]
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
            elif cfg.MODEL.BASE_MODEL.LAYER == 2:
                self.backbone = nn.Sequential(*list(res50_model.children())[:-4]) # output of layer2
                self.res_finetune = nn.Sequential(*list(res50_model.children())[-4:-2])
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
            else:
                self.backbone = nn.Sequential(*list(res50_model.children())[:-2]) # output of layer4: 2048x7x7
                self.res_finetune = nn.Identity()
                cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048

        if self.fusion_type == 'late':
            self.embed = TransformerEmbModel(cfg) # Regular Fusion Module
        elif self.fusion_type == 'smart':
            self.embed = MultiEntityTransformerEmbModel(cfg) # NEW: Multi-entity Temporal Fusion (MV-Former)
        else:
            print('WARNING: invalid setting for cfg.MODEL.EMBEDDER_MODEL.FUSION_TYPE:')
            print(self.fusion_type)
            exit(-1)

        # NEW - optionally include CLS token with smart tokens, additionally, can choose to only allow
        # gradients to prop to the backbone through the CLS token
        self.fuse_cls = False
        if 'FUSION_CLS' in cfg.MODEL.EMBEDDER_MODEL and cfg.MODEL.EMBEDDER_MODEL.FUSION_CLS == True:
            if self.backbone_type != 'timm' or self.fusion_type != 'smart':
                print('WARNING: Invalid config')
                print('FUSION_CLS can only be used with timm ViT backbones and smart fusion')
                exit(-1)
            else:
                print('FUSION_CLS enabled, cls token will be included with spatial tokens')
                self.fuse_cls = True
        self.cls_grad_only = False
        if 'CLS_GRAD_ONLY' in cfg.MODEL.EMBEDDER_MODEL and cfg.MODEL.EMBEDDER_MODEL.CLS_GRAD_ONLY == True:
            if not self.fuse_cls:
                print('WARNING: Invalid config')
                print('CLS_GRAD_ONLY can only be used with FUSION_CLS enabled')
                exit(-1)
            else:
                print('CLS_GRAD_ONLY enabled, gradients will only pass to the backbone through the CLS token')
                self.cls_grad_only = True

        self.embedding_size = self.embed.embedding_size
        
        if cfg.MODEL.PROJECTION:
            self.ssl_projection = MLPHead(cfg)
        if cfg.TRAINING_ALGO == 'classification':
            self.classifier = Classifier(cfg)

    def forward(self, x, num_frames=None, video_masks=None, project=False, classification=False):

        batch_size, num_steps, c, h, w = x.shape
        frames_per_batch = self.cfg.MODEL.BASE_MODEL.FRAMES_PER_BATCH
        num_blocks = int(math.ceil(float(num_steps)/frames_per_batch))
        backbone_out = []
        cls_backbone_out = [] # NEW: optionally also gather separate cls output
        cls_emb = None
        for i in range(num_blocks):
            curr_idx = i * frames_per_batch
            cur_steps = min(num_steps-curr_idx, frames_per_batch)
            curr_data = x[:, curr_idx:curr_idx+cur_steps]
            curr_data = curr_data.contiguous().view(-1, c, h, w)
            
            self.backbone.eval()
            with torch.no_grad():
                curr_emb = self.backbone(curr_data)
            curr_emb = self.res_finetune(curr_emb)

            if self.backbone_type == 'timm':
                if self.fusion_type == 'late' and self.late_type == 'cls':
                    # DINO CLS output
                    _, out_c = curr_emb.size()
                    out_h = 1
                    out_w = 1
                else:
                    # DINO last block output (with secondary cls output)
                    spc_emb, cls_emb = curr_emb
                    cls_backbone_out.append(cls_emb)
                    curr_emb = spc_emb
                    _, ntok, out_c = curr_emb.size()
                    curr_emb = torch.movedim(curr_emb,1,2)
                    curr_emb = curr_emb[:,:,1:] # remove cls token
                    out_h = int(math.sqrt(ntok-1)) # assuming square layout
                    out_w = out_h
                    curr_emb = curr_emb.reshape([curr_emb.size(0), out_c, out_h, out_w])            
            elif self.backbone_type == 'resnet':
                _, out_c, out_h, out_w = curr_emb.size()
            else:
                print('ERROR: unknown backbone type')
                exit(-1)
            curr_emb = curr_emb.contiguous().view(batch_size, cur_steps, out_c, out_h, out_w)
            backbone_out.append(curr_emb)

        if len(cls_backbone_out) > 0:
            cls_emb = torch.cat(cls_backbone_out, dim=0)
        x = torch.cat(backbone_out, dim=1)

        # NEW - Pass CLS embedding for optional extra uses
        if self.fusion_type == 'smart':
            x = self.embed(x, video_masks=video_masks, cls_emb=cls_emb)
        else:
            x = self.embed(x, video_masks=video_masks)

        if self.cfg.MODEL.PROJECTION and project:
            x = self.ssl_projection(x)
            x = F.normalize(x, dim=-1)
        elif self.cfg.MODEL.L2_NORMALIZE:
            x = F.normalize(x, dim=-1)
        if classification:
            return self.classifier(x)

        # NEW - optionally add direct CLS residual connection
        if self.use_cls_res:
            cls_res = self.cls_res_res(cls_emb)
            cls_res = cls_res.view(x.shape[0], x.shape[1], -1)
            if self.cfg.MODEL.L2_NORMALIZE:
                cls_res = F.normalize(cls_res, dim=-1)
            x += cls_res
            if self.cfg.MODEL.L2_NORMALIZE:
                x = F.normalize(x, dim=-1)

        return x



class TransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        assert cfg.MODEL.EMBEDDER_MODEL.FLATTEN_METHOD in ['max_pool','avg_pool']
        if cfg.MODEL.EMBEDDER_MODEL.FLATTEN_METHOD == 'max_pool':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.BatchNorm1d(channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        self.video_emb = nn.Linear(in_channels, hidden_channels)
        
        self.video_pos_enc = PositionalEncoder(cfg, hidden_channels, drop_rate, seq_len=cfg.TRAIN.NUM_FRAMES)
        if cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            self.video_encoder = Encoder(hidden_channels, drop_rate, cfg.MODEL.EMBEDDER_MODEL.NUM_HEADS, 
                                            cfg.MODEL.EMBEDDER_MODEL.D_FF, cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)
        
        self.embedding_layer = nn.Linear(hidden_channels, self.embedding_size)

    def forward(self, x, video_masks=None):
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)

        x = self.pooling(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.view(batch_size, num_steps, x.size(1))
        x = self.video_pos_enc(x)
        if self.cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            x = self.video_encoder(x, src_mask=video_masks)

        x = x.view(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x



# generic feature extactor wrapper for intermedia layers modified
# from: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
class FeatureExtractor(nn.Module):
    def __init__(self, model, layers, return_output=True):
        super().__init__()
        self.model = model
        self.layers = layers
        self.return_output = return_output
        self._features = {layer: torch.empty(0) for layer in layers}
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        out = self.model(x)
        # stack features
        all_f = []
        for l in self.layers:
            f = self._features[l]
            all_f.append(f)
        all_f = torch.concat(all_f, 2)
        if self.return_output:
            return all_f, out
        else:
            return all_f



# ViT Backbone Splitter:
# Code to split ViT Backbone foreward pass into two separate modules.
# Feed the initialized, pre-trained ViT into each sub-module, and execute with forward.
# Designed to work with TIMM ViT instances of VisionTransformer as implemented in
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class ViTFrontEnd(nn.Module):
    def __init__(self, model, nb):
        super().__init__()
        self.model = model
        self.nb = nb
        # pull "front" blocks
        self.blocks = []
        for i in range(len(self.model.blocks)):
            if i == nb: break
            self.blocks.append(self.model.blocks[i])
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        # based on forward_features (front-end)
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.blocks(x)
        return x


class ViTBackEnd(nn.Module):
    def __init__(self, model, nb, local_rank=None):
        super().__init__()
        if local_rank is None:
            local_rank = 0
        # pull settings
        self.global_pool = model.global_pool # string
        self.num_prefix_tokens = model.num_prefix_tokens # int
        # pull back blocks, make deepcopy to avoid crossed refs
        self.blocks = []
        for i in range(len(model.blocks)):
            if i >= nb:
                self.blocks.append(deepcopy(model.blocks[i]).cuda(local_rank))
        self.blocks = nn.Sequential(*self.blocks)
        # pull other layers
        self.norm = deepcopy(model.norm).cuda(local_rank)
        self.fc_norm = deepcopy(model.fc_norm).cuda(local_rank)
        self.head_drop = deepcopy(model.head_drop).cuda(local_rank)
        self.head = deepcopy(model.head).cuda(local_rank)

    def forward(self, x, pre_logits=False):
        x = self.blocks(x)
        x = self.norm(x)
        # forward_head:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)