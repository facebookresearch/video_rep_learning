# Code for MV-Former Modules, which integrate with transformer.py

# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *


# NEW: MV-Former Multi-entity Fusion Transformer module
class MultiEntityTransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print('Using Smart Pooling')
        self.cfg = cfg
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE

        # Number of channel for smart-pooling layers
        if 'SMART_POOL_CHANNELS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number for SMART_POOL_CHANNELS: 384')
            in_channels = 384
        else:
            in_channels = cfg.MODEL.EMBEDDER_MODEL.SMART_POOL_CHANNELS
        if 'VAL_PASS' in cfg.MODEL.EMBEDDER_MODEL and cfg.MODEL.EMBEDDER_MODEL.VAL_PASS:
            in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL

        # number of smart-pooling tokens
        if 'SMART_TOKENS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_TOKENS: 5')
            self.nst = 5
        else:
            self.nst = cfg.MODEL.EMBEDDER_MODEL.SMART_TOKENS
        if 'SMART_DYNAMIC_TOKENS' in cfg.MODEL.EMBEDDER_MODEL:
            self.nsdt = cfg.MODEL.EMBEDDER_MODEL.SMART_DYNAMIC_TOKENS
        else:
            self.nsdt = 0

        # NEW - smart-token-id one-hot (similar function to position embedding)
        self.one_hot_pos = "none"
        if "SMART_ONE_HOT" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.one_hot_pos = self.cfg.MODEL.EMBEDDER_MODEL.SMART_ONE_HOT
            assert self.one_hot_pos in ["none", "pool", "enc"]
        if self.one_hot_pos == "pool":
            # expand input to fc layers for one-hot channels
            in_channels += (self.nst + self.nsdt)

        # NEW - fixed width baseline: simulate the increased width of MTF without using multiple entities
        self.fwb = False
        if 'FIXED_WIDTH_BASELINE' in cfg.MODEL.EMBEDDER_MODEL:
            self.fwb = cfg.MODEL.EMBEDDER_MODEL.FIXED_WIDTH_BASELINE

        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = None
        if 'FC_LAYERS' in cfg.MODEL.EMBEDDER_MODEL:
            fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        
        if not self.fwb:
            self.pooling = LearnableTokenPooling(cfg)
        else:
            print('RUNNING FIXED WIDTH BASELINE')
            self.pooling = FWBPooling(cfg)

        # NEW - make fc_layers optional
        if fc_params is None:
            self.fc_layers = nn.Identity()
        else:
            self.fc_layers = []
            for channels, activate in fc_params:
                channels = channels*cap_scalar
                self.fc_layers.append(nn.Dropout(drop_rate))
                self.fc_layers.append(nn.Linear(in_channels, channels))
                self.fc_layers.append(nn.BatchNorm1d(channels))
                self.fc_layers.append(nn.ReLU(True))
                in_channels = channels
            self.fc_layers = nn.Sequential(*self.fc_layers)

        if self.one_hot_pos == "enc":
            hidden_channels -= self.nst

        self.video_emb = nn.Linear(in_channels, hidden_channels)
        
        self.video_pos_enc = PositionalEncoder(cfg, hidden_channels, drop_rate, seq_len=cfg.TRAIN.NUM_FRAMES)

        if self.one_hot_pos == "enc":
            hidden_channels += self.nst

        if cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            self.video_encoder = Encoder(hidden_channels, drop_rate, cfg.MODEL.EMBEDDER_MODEL.NUM_HEADS, 
                                            cfg.MODEL.EMBEDDER_MODEL.D_FF, cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)
        
        self.embedding_layer = nn.Linear(hidden_channels, self.embedding_size)

        # NEW - determine final pooling method
        if "SMART_FINAL" not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default smart final pooling: max pool')
            self.smart_final = 'max'
        else:
            self.smart_final = cfg.MODEL.EMBEDDER_MODEL.SMART_FINAL
            assert self.smart_final in ['max','one','avg','lin']
        if self.smart_final == 'lin':
            # linear reduction layer
            tt = self.nst + self.nsdt
            self.lin_final = nn.Linear(tt*hidden_channels, hidden_channels)

        # NEW - backbone warmup: only start fine-tuning the backbone after a certain number of epochs
        self.in_backbone_warmup = False
        if "BACKBONE_WARMUP" in self.cfg.TRAIN:
            self.in_backbone_warmup = True
            print('BACKBONE_WARMUP Enabled, up to epoch ' + str(self.cfg.TRAIN.BACKBONE_WARMUP))



    # warmup contol switch
    def set_warmup_status(self, new_status):
        self.in_backbone_warmup = new_status
        print('DEBUG - BACKBONE WARMUP STATUS SET TO: ' + str(self.in_backbone_warmup))
        return



    # NEW - optionally append cls_emb to other smart tokens, if provided
    def forward(self, x, video_masks=None, cls_emb=None):
        
        # WARM-UP: stop fine-tuning of backbone during warmup period
        if self.in_backbone_warmup:
            x = x.detach()

        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)

        x = self.pooling(x, cls_emb, batch_size)

        _, c, ntok = x.shape # new c after pooling
        x = x.view(batch_size*num_steps, c, ntok)
        x = torch.movedim(x,2,1)

        # ONE-HOT OPTION 1: appended directly after smart token pooling
        if self.one_hot_pos == "pool":
            one_hot = torch.eye(ntok, device=x.get_device())
            one_hot = torch.unsqueeze(one_hot, 0)
            one_hot = torch.cat([one_hot]*x.shape[0],0)
            x = torch.cat([x, one_hot], 2)
            c = x.shape[-1]

        x = x.reshape(batch_size*num_steps*ntok, c)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.reshape(batch_size*num_steps, ntok, x.size(1))
        x = x.view(batch_size, num_steps, ntok, x.size(2))
        x = torch.movedim(x,2,1)
        x = x.reshape([batch_size*ntok, num_steps, x.size(3)])
        x = self.video_pos_enc(x)
        x = x.reshape([batch_size, ntok, num_steps, x.size(2)])

        # ONE-HOT OPTION 2: appended directly before video encoder
        if self.one_hot_pos == "enc":
            one_hot = torch.eye(ntok, device=x.get_device())
            one_hot = torch.unsqueeze(one_hot, 0)
            one_hot = torch.unsqueeze(one_hot, 2)
            one_hot = torch.cat([one_hot]*x.shape[0],0)
            one_hot = torch.cat([one_hot]*x.shape[2],2)
            x = torch.cat([x, one_hot], 3)

        x = x.reshape([batch_size, ntok*num_steps, x.size(3)])

        if self.cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            # expand the mask
            vm = video_masks
            if vm is not None:
                vm = torch.stack([vm]*(ntok), dim=2)
                vm = vm.reshape([batch_size, 1, ntok*num_steps])
            x = self.video_encoder(x, src_mask=vm)                
        x = x.view([batch_size, ntok, num_steps, x.size(2)])

        # Final reduction to frame-level representation:
        if self.smart_final == 'max':
            # OPTION 1: Max pool out the token dimension
            x, _ = torch.max(x, dim=1)
        elif self.smart_final == 'one':
            # OPTION 2: Take one token "CLS-style"
            x = x[:,0,:,:]
        elif self.smart_final == 'avg':
            # OPTION 3: average pool the token dimension
            x = torch.mean(x, dim=1)
        elif self.smart_final == 'lin':
            # OPTION 4: linear layer reduction
            x = torch.movedim(x,1,2)
            x = torch.reshape(x, [batch_size, num_steps, -1])
            x = self.lin_final(x)

        x = x.reshape(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x




# NEW: Learnable spatial token pooling module with cross-attention and
# learned queries for use with Multi-Entity Fusion
class LearnableTokenPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if 'SMART_TOKENS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_TOKENS: 5')
            self.nst = 5
        else:
            self.nst = cfg.MODEL.EMBEDDER_MODEL.SMART_TOKENS
        
        if 'SMART_DYNAMIC_TOKENS' in cfg.MODEL.EMBEDDER_MODEL:
            self.nsdt = cfg.MODEL.EMBEDDER_MODEL.SMART_DYNAMIC_TOKENS
        else:
            self.nsdt = 0

        if 'SMART_POOL_CHANNELS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_POOL_CHANNELS: 384')
            self.spc = 384
        else:
            self.spc = cfg.MODEL.EMBEDDER_MODEL.SMART_POOL_CHANNELS
        self.in_c = cfg.MODEL.BASE_MODEL.OUT_CHANNEL

        # identify d_dyn_in
        d_dyn_in = self.in_c
        if 'SMART_FEATS' in cfg.MODEL.EMBEDDER_MODEL:
            sfl = str(cfg.MODEL.EMBEDDER_MODEL.SMART_FEATS)
            if ',' in sfl: # multiple-layer extraction mode
                sfl_n = len(sfl.split(','))
                d_dyn_in = int(d_dyn_in/sfl_n)

        self.cross_att = LSTPCrossAtt(cfg=self.cfg, num_static=self.nst, num_dynamic=self.nsdt, 
            d_model_K=self.in_c, d_model_V=self.in_c, d_model=self.spc, d_dyn_in=d_dyn_in)


    # INPUT: [B, C, H, W]
    # OUTPUT: [B, Cn, T]
    def forward(self, x, dyn_in, batch_size):
        bn, cn, hn, wn = x.shape
        x = x.view([bn, cn, hn*wn])
        x = torch.movedim(x, 2, 1)
        
        # handle the views separately
        _, nt, nc = x.shape
        x = x.view(batch_size, -1, nt, nc)
        if self.nsdt > 0:
            _, ncc = dyn_in.shape
            dyn_in = dyn_in.view(batch_size, -1, ncc)
        x_out = []
        for i in range(batch_size):
            x_c = x[i,...]
            if self.nsdt > 0:
                d_c = dyn_in[i,...]
            else:
                d_c = None
            x_c = self.cross_att(x_c, x_c, d_c)
            x_c = x_c[:,0,:,:]
            x_c = torch.movedim(x_c, 2, 1)
            x_out.append(x_c)
        x = torch.cat(x_out, 0)
        return x



'''
Cross Attention for use with Learnable Token Pooling. Supports two types of queries
1) static queries, fixed for all inputs
2) dynamic queries, conditioned on an extra input
'''
class LSTPCrossAtt(nn.Module):
    def __init__(self, cfg, num_static, num_dynamic, d_model_K, d_model_V, d_model, d_dyn_in=None, dout_p=0.0):
        super(LSTPCrossAtt, self).__init__()
        self.cfg = cfg
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.d_model = d_model
        
        # NEW - OPTIONAL - direct pass through of the base feature vector as the value vector
        self.pass_through = False
        if "VAL_PASS" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.pass_through = self.cfg.MODEL.EMBEDDER_MODEL.VAL_PASS

        # NEW - OPTIONAL: Smart disjoint attention, enforces disjoint attention through max attention mechanism
        self.disjoint_att = False
        if "SMART_DISJOINT" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.disjoint_att = self.cfg.MODEL.EMBEDDER_MODEL.SMART_DISJOINT
            print('SMART_DISJOINT ENABLED')

        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        if self.pass_through:
            self.linear_V2d = nn.Identity()
        else:
            self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)            

        # NEW - Optional Layernorm on Keys before Attention based on:
        # https://aclanthology.org/2023.findings-acl.895.pdf
        self.ln_keys = False
        if "SMART_LN_KEYS" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.ln_keys = self.cfg.MODEL.EMBEDDER_MODEL.SMART_LN_KEYS

        # NEW - Dynamic Token control method:
        #   separate (default) - the dynamic token queries for each frame are generated separately for each frames CLS features 
        #   first - dynamic queries are derived from first frame CLS features and shared with all other frames
        #   average - dynamic queries are shared by all and derived from average of all CLS features
        self.dyn_ctrl = "separate"
        if "DYNAMIC_CTRL" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.dyn_ctrl = self.cfg.MODEL.EMBEDDER_MODEL.DYNAMIC_CTRL
            assert self.dyn_ctrl in ["separate", "first", "average"]

        if num_static == 0 and num_dynamic == 0:
            print('ERROR: cannot have both num_static == 0 and num_dynamic == 0')
            exit(-1)

        # static smart token queries
        self.num_s = num_static
        self.stat = False
        if self.num_s > 0:
            self.stat = True
            self.Q_s = nn.Parameter(torch.empty([1, self.num_s, self.d_model], dtype=torch.float32))
            nn.init.kaiming_uniform_(self.Q_s, a=math.sqrt(5)) # same as nn.Linear weight
            self.Q_s_b = nn.Parameter(torch.empty(self.d_model, dtype=torch.float32))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Q_s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.Q_s_b, -bound, bound)

        # dynamic smart token query generation
        self.num_d = num_dynamic
        self.dyn = False
        if self.num_d > 0:
            self.dyn = True
            self.d_dyn_in = d_dyn_in
            if d_dyn_in is None:
                print('LSTPCrossAtt - WARNING: d_dyn_in in not set, defaulting to equal d_model_V')
                self.d_dyn_in = self.d_model_V
            self.in2dynQ = nn.Linear(self.d_dyn_in, self.d_model*self.num_d)

        # optional dropout
        self.dout_p = dout_p
        self.dropout = nn.Dropout(self.dout_p)
        
        # visualizaing attention maps:
        self.visual = True
        self.attn_holder = nn.Identity()


    
    def forward(self, K, V, dyn_in=None, mask=None):
        ''' 
            K, V: (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
        '''
        B, _, _ = V.shape
        # (B, Sm, D) <- (B, Sm, Dm)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        # Prepare dynamic token queries
        if self.dyn:
            assert dyn_in is not None
            if self.dyn_ctrl == 'separate':
                Bdi = B
            elif self.dyn_ctrl == 'first':
                dyn_in = torch.unsqueeze(dyn_in[0,:],0)
                Bdi = 1
            else: # average
                dyn_in = torch.mean(dyn_in, 0, keepdim=True)
                Bdi = 1
            Q_d = self.in2dynQ(dyn_in)
            Q_d = Q_d.view(Bdi, self.num_d, self.d_model)

        # Combine static and dynamic tokens
        if not self.stat: # only dynamic tokens
            Q = Q_d
            Bq = Bdi
        elif not self.dyn: # only static tokens
            Q = self.Q_s + self.Q_s_b
            Bq = 1
        else: # both static and dynamic tokens
            Q_temp = self.Q_s + self.Q_s_b
            if self.dyn_ctrl == 'separate':
                Q_s_B = torch.cat([Q_temp]*B, 0)
            else:
                Q_s_B = Q_temp
            Q = torch.cat([Q_s_B, Q_d], 1)
            Bq = Bdi

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(Bq, -1, 1, self.d_model).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, 1, self.d_model).transpose(-3, -2)
        V = V.view(B, -1, 1, V.shape[-1]).transpose(-3, -2)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        # Optional: apply layer norm to keys
        if self.ln_keys:
            K = F.normalize(K, dim=-1)

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        if self.visual:
            ret, self.attn_matrix = attention(Q, K, V, mask, self.dropout, self.visual, disjoint=self.disjoint_att)
            self.attn_matrix = self.attn_matrix.mean(-3)
            _ = self.attn_holder(self.attn_matrix)
        else:
            ret = attention(Q, K, V, mask, self.dropout, disjoint=self.disjoint_att)
        return ret




# NEW: Alternative pooling method for the Fixed-Width Baseline. Simulate the increased width of MTF without
# actually using multiple entities...
class FWBPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if 'SMART_TOKENS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_TOKENS: 5')
            self.nst = 5
        else:
            self.nst = cfg.MODEL.EMBEDDER_MODEL.SMART_TOKENS
        
        if 'SMART_DYNAMIC_TOKENS' in cfg.MODEL.EMBEDDER_MODEL:
            self.nsdt = cfg.MODEL.EMBEDDER_MODEL.SMART_DYNAMIC_TOKENS
        else:
            self.nsdt = 0

        if 'SMART_POOL_CHANNELS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_POOL_CHANNELS: 384')
            self.spc = 384
        else:
            self.spc = cfg.MODEL.EMBEDDER_MODEL.SMART_POOL_CHANNELS
        self.in_c = cfg.MODEL.BASE_MODEL.OUT_CHANNEL

        # identify d_dyn_in
        d_dyn_in = self.in_c
        if 'SMART_FEATS' in cfg.MODEL.EMBEDDER_MODEL:
            sfl = str(cfg.MODEL.EMBEDDER_MODEL.SMART_FEATS)
            if ',' in sfl: # multiple-layer extraction mode
                sfl_n = len(sfl.split(','))
                d_dyn_in = int(d_dyn_in/sfl_n)

        # linear conversion layer
        tt = self.nst + self.nsdt
        self.lin_conv = nn.Linear(d_dyn_in, self.spc * tt)


    # cls_in = CLS token
    # OUTPUT: [B, Cn, T]
    def forward(self, x, cls_in, batch_size):
        bn, cn, hn, wn = x.shape
        tt = self.nst + self.nsdt
        x = self.lin_conv(cls_in)
        x = x.reshape([bn, -1, tt])
        return x