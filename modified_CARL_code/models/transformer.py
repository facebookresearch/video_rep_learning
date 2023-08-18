import math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.resnet_c2d import *

def attention(Q, K, V, mask=None, dropout=None, visual=False):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)

    d_k = Q.size(-1)
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    if dropout is not None:
        out = dropout(out)

    # (B, *(H), seq_len, d_model//H = d_k)
    if visual:
        return out, softmax.detach()
    else:
        return out


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None, d_out=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_out = d_out
        if self.d_out is None:
            self.d_out = self.d_model_Q

        if self.d_model is None:
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_out)

        self.dropout = nn.Dropout(self.dout_p)
        self.visual = False

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask=None):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
        '''
        B, Sq, d_model_Q = Q.shape
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        if self.visual:
            Q, self.attn_matrix = attention(Q, K, V, mask, self.dropout, self.visual)
            self.attn_matrix = self.attn_matrix.mean(-3)
        else:
            Q = attention(Q, K, V, mask, self.dropout)
        # (B, Sq, D) <- (B, H, Sq, d_k)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        # (B, Sq, Dq)
        Q = self.linear_d2Q(Q)

        return Q

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def generate_sincos_embedding(seq_len, d_model, train_len=None):
    odds = np.arange(0, d_model, 2)
    evens = np.arange(1, d_model, 2)
    pos_enc_mat = np.zeros((seq_len, d_model))
    if train_len is None:
        pos_list = np.arange(seq_len)
    else:
        pos_list = np.linspace(0, train_len-1, num=seq_len)

    for i, pos in enumerate(pos_list):
        pos_enc_mat[i, odds] = np.sin(pos / (10000 ** (odds / d_model)))
        pos_enc_mat[i, evens] = np.cos(pos / (10000 ** (evens / d_model)))

    return torch.from_numpy(pos_enc_mat).unsqueeze(0)

class PositionalEncoder(nn.Module):
    def __init__(self, cfg, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        self.seq_len = seq_len

    def forward(self, x):
        B, S, d_model = x.shape
        if S != self.seq_len:
            pos_enc_mat = generate_sincos_embedding(S, d_model, self.seq_len)
            x = x + pos_enc_mat.type_as(x)
        else:
            pos_enc_mat = generate_sincos_embedding(S, d_model)
            x = x + pos_enc_mat.type_as(x)
        x = self.dropout(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer): 
        # x (B, S, D)
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)

        return x + res

class BridgeConnection(nn.Module):
    def __init__(self, in_dim, out_dim, dout_p):
        super(BridgeConnection, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return self.activation(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dout_p = dout_p
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        '''In, Out: (B, S, D)'''
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H=8, d_ff=None, d_hidden=None):
        super(EncoderLayer, self).__init__()
        self.res_layer0 = ResidualConnection(d_model, dout_p)
        self.res_layer1 = ResidualConnection(d_model, dout_p)
        if d_hidden is None: d_hidden = d_model
        if d_ff is None: d_ff = 4*d_model
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H, d_model=d_hidden)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, src_mask=None):
        '''
        in:
            x: (B, S, d_model), src_mask: (B, 1, S)
        out:
            (B, S, d_model)
        '''
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layer0(x, sublayer0)
        x = self.res_layer1(x, sublayer1)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, dout_p, H, d_ff, N, d_hidden=None):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff, d_hidden), N)
        
    def forward(self, x, src_mask=None):
        '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
        '''
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x



# ==================================================
# ViT Backbone Splitter
# ==================================================

# EXPERIMENTAL - Code to split ViT Backbone foreward pass into two separate modules
# for efficient training
# Feed the initialized, pre-trained ViT into each sub-module, and execute with forward
# Designed to work with TIMM ViT instances of VisionTransformer as implemented in
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py



# original versions with overlapping layer refs:


# nb = number of block to stop at
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
            # print('=== %i ==='%i)
            # print(self.model.blocks[i])
            # TODO - clean up
        # DEBUG
        # print('FRONT END BLOCKS:')
        # print(len(self.my_blocks))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        # based on forward_features (front-end)
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        # checkpoint_seq not supported:
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        x = self.blocks(x)
        # for blk in self.my_blocks:
        #     x = blk(x)
        return x



# class ViTBackEnd(nn.Module):
#     def __init__(self, model, nb):
#         super().__init__()
#         self.model = model
#         self.nb = nb
#         # pull "back" blocks
#         self.my_blocks = []
#         for i in range(len(self.model.blocks)):
#             if i >= nb:
#                 self.my_blocks.append(self.model.blocks[i])
#             # print('=== %i ==='%i)
#             # print(self.model.blocks[i])
#             # TODO - clean up
#         # DEBUG
#         # print('FRONT END BLOCKS:')
#         # print(len(self.my_blocks))
#         # self.my_blocks = nn.Sequential(*self.my_blocks)

#     def forward(self, x):
#         # based on forward_features (back-end)
#         # x = self.my_blocks(x)
#         for blk in self.my_blocks:
#             x = blk(x)
#         x = self.model.norm(x)
#         x = self.model.forward_head(x)
#         return x



# version that have no overlapping layer refs:



# # nb = number of block to stop at
# class ViTFrontEnd(nn.Module):
#     def __init__(self, model, nb):
#         super().__init__()
#         self.model = model
#         self.nb = nb
#         # pull "front" blocks
#         self.my_blocks = []
#         for i in range(len(self.model.blocks)):
#             if i == nb: break
#             self.my_blocks.append(self.model.blocks[i])
#             # print('=== %i ==='%i)
#             # print(self.model.blocks[i])
#             # TODO - clean up
#         # DEBUG
#         # print('FRONT END BLOCKS:')
#         # print(len(self.my_blocks))
#         # self.my_blocks = nn.Sequential(*self.my_blocks)
#         # pull elements
#         self.patch_embed = model.patch_embed
#         self.no_embed_class = model.no_embed_class
#         self.cls_token = model.cls_token
#         self.pos_embed = model.pos_embed
#         self.pos_drop = model.pos_drop
#         self.patch_drop = model.patch_drop
#         self.norm_pre = model.norm_pre

#     def forward(self, x):
#         # based on forward_features (front-end)
#         # x = self.model.patch_embed(x)
#         # x = self.model._pos_embed(x)
#         # x = self.model.patch_drop(x)
#         # x = self.model.norm_pre(x)

#         x = self.patch_embed(x)
#         # _pos_embed:
#         if self.no_embed_class:
#             # deit-3, updated JAX (big vision)
#             # position embedding does not overlap with class token, add then concat
#             x = x + self.pos_embed
#             if self.cls_token is not None:
#                 x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
#         else:
#             # original timm, JAX, and deit vit impl
#             # pos_embed has entry for class token, concat then add
#             if self.cls_token is not None:
#                 x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
#             x = x + self.pos_embed
#         x = self.pos_drop(x)
#         x = self.patch_drop(x)
#         x = self.norm_pre(x)

#         # checkpoint_seq not supported:
#         # if self.grad_checkpointing and not torch.jit.is_scripting():
#         #     x = checkpoint_seq(self.blocks, x)
#         # else:
#         #     x = self.blocks(x)
#         # x = self.my_blocks(x)
#         for blk in self.my_blocks:
#             x = blk(x)
#         return x




# class ViTBackEnd(nn.Module):
#     def __init__(self, model, nb):
#         super().__init__()
#         # self.model = model
#         # pull "back" blocks
#         self.my_blocks = []
#         for i in range(len(model.blocks)):
#             if i >= nb:
#                 self.my_blocks.append(model.blocks[i])
#             # print('=== %i ==='%i)
#             # print(self.model.blocks[i])
#             # TODO - clean up
#         # DEBUG
#         # print('FRONT END BLOCKS:')
#         # print(len(self.my_blocks))
#         # self.my_blocks = nn.Sequential(*self.my_blocks)
#         # pull elements
#         self.norm = model.norm
#         self.global_pool = model.global_pool
#         self.num_prefix_tokens = model.num_prefix_tokens
#         self.fc_norm = model.fc_norm
#         self.head_drop = model.head_drop
#         self.head = model.head

#     def forward(self, x, pre_logits=False):
#         # based on forward_features (back-end)
#         # x = self.my_blocks(x)
#         for blk in self.my_blocks:
#             x = blk(x)
#         x = self.norm(x)
        
#         # x = self.model.forward_head(x)
#         # return x
#         # forward_head:
#         if self.global_pool:
#             x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#         x = self.fc_norm(x)
#         x = self.head_drop(x)
#         return x if pre_logits else self.head(x)



# pull back-end ops and deepcopy
class ViTBackEnd(nn.Module):
    def __init__(self, model, nb, local_rank=None):
        super().__init__()
        if local_rank is None:
            local_rank = 0
        # pull settings
        self.global_pool = model.global_pool # string
        self.num_prefix_tokens = model.num_prefix_tokens # int
        # pull back blocks
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
        # forward_features (back-end):
        # for blk in self.my_blocks:
        #     x = blk(x)
        x = self.blocks(x)
        x = self.norm(x)
        # forward_head:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)



# # DEBUG - TODO - TEMP - REMOVE
# # Verify that the two halves join up correctly...
# class ViTTestWrap(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.front = ViTFrontEnd(model, 9)
#         self.back = ViTBackEnd(model, 9)

#     def forward(self, x):
#         x = self.front(x)
#         x = self.back(x)
#         print('ViTTestWrap forward')
#         return x



# ==================================================
# OVERALL MODEL
# ==================================================



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

        # NEW - optional CLS_RES connection (only for use with Smart Token Pooling)
        self.use_cls_res = False
        if 'CLS_RES' in self.cfg.MODEL and self.cfg.MODEL.CLS_RES:
            self.use_cls_res = True
            if self.fusion_type == 'late':
                print('ERROR: CLS_RES cannot be used with late fusion')
                exit(-1)

        # NEW - experimental "headless" mode with backbone only, no temporal fusion
        self.headless_mode = False
        if 'HEADLESS_MODE' in self.cfg.MODEL:
            self.headless_mode = self.cfg.MODEL.HEADLESS_MODE

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
            
            # TODO - remove this
            # partial fine-tuning, currently supported for DINO v1:
            # if model_name in ['vit_small_patch16_224.dino', 'vit_small_patch8_224.dino', 'vit_base_patch16_224.dino', 'vit_base_patch8_224.dino']:
                
            # NEW - EXPERIMENTAL - OPTIONAL RESIDUAL CONNECTION FOR CLS OUTPUT
            if self.use_cls_res:
                self.cls_res_res = nn.Linear(cfg.MODEL.BASE_MODEL.OUT_CHANNEL, cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)

            # Select layers to extract and stack if selecting multiple layers.
            # Setting is specified as either as a single block number or a comma-separated list of block numbers (i.e. "9,10,11")
            if self.fusion_type != 'late':
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

            # fully frozen vs partial finetuning
            if cfg.MODEL.BASE_MODEL.LAYER < 0 or cfg.MODEL.BASE_MODEL.LAYER >= blk_count:
                # fully frozen
                print('backbone fully frozen')
                if self.fusion_type != 'late':
                    model = FeatureExtractor(model, extract_ids)
                self.backbone = model
                self.res_finetune = nn.Identity()
            
            # ORIGINAL SPLITTING METHOD, VERIFIED:
            # else:
            #     freeze_vit(model, cfg.MODEL.BASE_MODEL.LAYER)
            #     if self.fusion_type != 'late':
            #         model = FeatureExtractor(model, extract_ids)
            #     self.backbone = nn.Identity()
            #     self.res_finetune = model

            # EXPERIMENTAL SPLITTING METHOD:
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

            # TODO - remove this
            # else:
            #     # TODO - add support for other TIMM model types
                
            #     if self.fusion_type != 'late':
            #         print('ERROR: fusion type %s not supported for: %s'%(self.fusion_type, model_name))
            #         exit(-1)
            #     self.backbone = model
            #     self.res_finetune = nn.Identity()
            #     print('WARNING: partial backbone fine-tuning not currently supported for: %s'%model_name)

        else: # RESNET Backbones (original CARL)
            self.backbone_type = 'resnet'
            res50_model = models.resnet50(pretrained=True)
            # NEW - modification - missing model loading?
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
        
        if self.headless_mode:
            return

        if self.fusion_type == 'late':
            self.embed = TransformerEmbModel(cfg)
        elif self.fusion_type == 'early':
            self.embed = EarlyFusionTransformerEmbModel(cfg)
        elif self.fusion_type == 'smart':
            # self.embed = SmartPoolingTransformerEmbModel(cfg)
            # TODO - EXPERIMENTAL - NEW VERSION
            self.embed = SmartPoolingTransformerEmbModelV2(cfg)
        else:
            print('WARNING: invalid setting for cfg.MODEL.EMBEDDER_MODEL.FUSION_TYPE:')
            print(self.fusion_type)
            exit(-1)

        # NEW - optionally include CLS token with smart tokens, additionally, can choose to only allow
        # gradients to the backbone through the CLS token
        # TODO - DEPRECATE AND REMOVE
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
        # TODO - reduce redundancy

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
                if self.fusion_type == 'late':
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
                    curr_emb = curr_emb[:,:,1:] # remove cls token # TODO - maybe keep CLS token for additional uses
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

        # NEW - EXPERIMENTAL - "HEADLESS" Baseline (backbone only)
        if self.headless_mode:
            cls_emb = cls_emb.view(x.shape[0], x.shape[1], -1)
            if self.cfg.MODEL.L2_NORMALIZE:
                cls_emb = F.normalize(cls_emb, dim=-1)
            return cls_emb
            
        # TODO - Deprecate and Remove this:
        # NEW - optionally append the CLS token to the spatial tokens, and optionally pass gradients only through CLS
        # if self.fuse_cls:
        #     x = self.embed(x, video_masks=video_masks, cls_emb=cls_emb, cls_grad_only=self.cls_grad_only)
        # else:
        #     x = self.embed(x, video_masks=video_masks)

        # NEW - Pass cls embedding for optional extra uses
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



# NEW: helper to freeze the parameters of a timm-loaded vit model up to a specified layer
def freeze_vit(model, layer, silent=False):
    fc = 0
    fb = 0
    for i,c in enumerate(model.children()):
        if i < 4: # input layers
            for p in c.parameters():
                p.requires_grad = False
                fc += 1
        else: # transformer blocks
            for b_idx, b in enumerate(c.children()):
                if b_idx == layer: break
                for p in b.parameters():
                    p.requires_grad = False
                    fc += 1
                fb += 1
            break
    if not silent:
        print('frozen block count: ' + str(fb))
        print('frozen param count: ' + str(fc))



# ==================================================
# TRANSFORMER-BASED TEMPORAL FUSION MODULES
# ==================================================



# Original fusion module  from CARL:
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
        self.pooling = nn.AdaptiveMaxPool2d(1)
        
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



# ==================================================



# NEW: Early-fusion transformer fusion module, passes more spatial tokens to the fusion module
# currently only supports a ViT-style backbone
# NOTE: This module is partially deprecated and not fully supported
class EarlyFusionTransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        
        # TODO - control the pooling size
        # self.pooling = nn.AdaptiveMaxPool2d(1)
        self.pooling = nn.AdaptiveMaxPool2d(3)

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
        note_print(x, 1)
        # 1: torch.Size([8, 80, 2048, 7, 7])
        x = x.view(batch_size*num_steps, c, h, w)
        note_print(x, 2)
        # 2: torch.Size([640, 2048, 7, 7])
        x = self.pooling(x)
        note_print(x, 3)
        # 3: torch.Size([640, 2048, 3, 3])
        _, _, hn, wn = x.shape
        x = x.view(batch_size*num_steps, c, hn*wn)
        note_print(x, 3.1)
        # 3.1: torch.Size([640, 2048, 9])
        x = torch.movedim(x,2,1)
        note_print(x, 3.2)
        # 3.2: torch.Size([640, 9, 2048])
        x = x.reshape(batch_size*num_steps*hn*wn, c)
        # x = torch.flatten(x, start_dim=1)
        note_print(x, 4)
        # 4: torch.Size([5760, 2048])
        x = self.fc_layers(x)
        note_print(x, 5)
        # 5: torch.Size([5760, 512])
        x = self.video_emb(x)
        note_print(x, 6)
        # 6: torch.Size([5760, 256])
        x = x.reshape(batch_size*num_steps, hn*wn, x.size(1))
        note_print(x, 6.1)
        # 6.1: torch.Size([640, 9, 256])
        x = x.view(batch_size, num_steps, hn*wn, x.size(2))
        note_print(x, 6.2)
        # 6.2: torch.Size([8, 80, 9, 256])
        # x = x.view(batch_size, num_steps, x.size(1))
        # note_print(x, 7)
        
        x = torch.movedim(x,2,1)
        note_print(x, 6.3)
        # 6.3: torch.Size([8, 9, 80, 256])
        x = x.reshape([batch_size*hn*wn, num_steps, x.size(3)])
        note_print(x, 6.4)
        # 6.4: torch.Size([72, 80, 256])
        x = self.video_pos_enc(x)
        note_print(x, 8)
        # 8: torch.Size([72, 80, 256])
        x = x.reshape([batch_size, hn*wn, num_steps, x.size(2)])
        note_print(x, 8.1)
        x = x.reshape([batch_size, hn*wn*num_steps, x.size(3)])
        note_print(x, 8.2)
        # TODO - add SPATIAL POSITION EMBEDDING somewhere here?
        if self.cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            # expand the mask
            vm = video_masks
            if vm is not None:
                note_print(vm, 99)
                # 99: torch.Size([8, 1, 80])
                vm = torch.stack([vm]*(hn*wn), dim=2)
                note_print(vm, 99.1)
                # 99.1: torch.Size([8, 1, 9, 80])
                vm = vm.reshape([batch_size, 1, hn*wn*num_steps])
                note_print(vm, 99.2)
                # 99.2: torch.Size([8, 1, 720])
            x = self.video_encoder(x, src_mask=vm)
        note_print(x, 9)
        # 9: torch.Size([8, 720, 256])
        # NEW: Max pool out the token dimension
        x = x.view([batch_size, hn*wn, num_steps, x.size(2)])
        note_print(x, 9.1)
        x, _ = torch.max(x, dim=1)
        note_print(x, 9.2)
        x = x.view(batch_size*num_steps, -1)
        note_print(x, 10)
        x = self.embedding_layer(x)
        note_print(x, 11)
        x = x.view(batch_size, num_steps, self.embedding_size)
        note_print(x, 12)
        # TODO - clean up
        # exit()
        return x



# ==================================================



# NEW: Fusion module with smart spatial token pooling
class SmartPoolingTransformerEmbModel(nn.Module):
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

        # number of smart-pooling tokens
        if 'SMART_TOKENS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_TOKENS: 5')
            self.nst = 5
        else:
            self.nst = cfg.MODEL.EMBEDDER_MODEL.SMART_TOKENS

        # NEW - smart-token-id one-hot (similar function to position embedding)
        self.one_hot_pos = "none"
        if "SMART_ONE_HOT" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.one_hot_pos = self.cfg.MODEL.EMBEDDER_MODEL.SMART_ONE_HOT
            assert self.one_hot_pos in ["none", "pool", "enc"]
        if self.one_hot_pos == "pool":
            # expand input to fc layers for one-hot channels
            in_channels += self.nst

        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = None
        if 'FC_LAYERS' in cfg.MODEL.EMBEDDER_MODEL:
            fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        
        self.pooling = SmartPooling(cfg)

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
            assert self.smart_final in ['max','one']

        # NEW - backbone warmup: only start fine-tuning the backbone after a certain number of epochs
        self.in_backbone_warmup = False
        if "BACKBONE_WARMUP" in self.cfg.TRAIN:
            self.in_backbone_warmup = True
            print('BACKBONE_WARMUP Enabled, up to epoch ' + str(self.cfg.TRAIN.BACKBONE_WARMUP))



    # TODO EXPERIMENTAL - warmup contol switch
    def set_warmup_status(self, new_status):
        self.in_backbone_warmup = new_status
        print('DEBUG - BACKBONE WARMUP STATUS SET TO: ' + str(self.in_backbone_warmup))
        return



    # NEW - experimental, optionally append cls_emb to other smart tokens, if provided
    def forward(self, x, video_masks=None, cls_emb=None):
        
        # OPTIONAL: only permit gradients through the cls token
        # TODO - deprecate this
        # if cls_emb is not None and cls_grad_only:
        #     x = x.detach()

        # WARM-UP: stop fine-tuning of backbone during warmup period
        if self.in_backbone_warmup:
            x = x.detach()

        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)
        x = self.pooling(x)
        _, c, ntok = x.shape # new c after pooling
        x = x.view(batch_size*num_steps, c, ntok)
        x = torch.movedim(x,2,1)

        # OPTIONAL: append cls token along with other tokens
        # TODO - deprecate this
        # if cls_emb is not None:
        #     cls_emb = torch.unsqueeze(cls_emb,1)
        #     x = torch.cat([cls_emb, x], 1)
        #     ntok += 1

        # OPTION 1: One-Hot appended directly after smart token pooling
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

        # OPTION 2: One-Hot appended directly before video encoder
        if self.one_hot_pos == "enc":
            one_hot = torch.eye(ntok, device=x.get_device())
            one_hot = torch.unsqueeze(one_hot, 0)
            one_hot = torch.unsqueeze(one_hot, 2)
            one_hot = torch.cat([one_hot]*x.shape[0],0)
            one_hot = torch.cat([one_hot]*x.shape[2],2)
            x = torch.cat([x, one_hot], 3)

        x = x.reshape([batch_size, ntok*num_steps, x.size(3)])

        # TODO - add a SPATIAL POSITION EMBEDDING

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

        x = x.reshape(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x




# NEW: Smart pooling module with cross-attention and learned queries
# for use with SmartPoolingTransformerEmbModel
class SmartPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if 'SMART_TOKENS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_TOKENS: 5')
            self.nst = 5
        else:
            self.nst = cfg.MODEL.EMBEDDER_MODEL.SMART_TOKENS
        if 'SMART_POOL_CHANNELS' not in cfg.MODEL.EMBEDDER_MODEL:
            print('Using default number of SMART_POOL_CHANNELS: 384')
            self.spc = 384 # TODO - need to set this match the CLS dimension
        else:
            self.spc = cfg.MODEL.EMBEDDER_MODEL.SMART_POOL_CHANNELS
        self.in_c = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        self.cross_att = SmartCrossAttention(self.nst, self.in_c, self.in_c, 1, d_model=self.spc)
        self.dummy = None # dummy input placeholder


    def make_dummy(self, x):
        self.dummy = torch.eye(self.nst, device=x.get_device())
        self.dummy = torch.unsqueeze(self.dummy, 0)


    # INPUT: [B, C, H, W]
    # OUTPUT: [B, Cn, T]
    def forward(self, x):
        if self.dummy is None:
            self.make_dummy(x)
        bn, cn, hn, wn = x.shape
        x = x.view([bn, cn, hn*wn])
        x = torch.movedim(x, 2, 1)
        x = self.cross_att(self.dummy, x, x)
        x = x[:,0,:,:]
        x = torch.movedim(x, 2, 1)
        return x




# NEW - modified version of MultiheadedAttention for smart token pooling
# for use with SmartPooling
class SmartCrossAttention(nn.Module):
    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None, d_out=None):
        super(SmartCrossAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_out = d_out
        if self.d_out is None:
            self.d_out = self.d_model_Q

        if self.d_model is None:
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        # self.linear_d2Q = nn.Linear(self.d_model, self.d_out)
        # self.linear_V2d = nn.Identity() # direct pass through of V

        self.dropout = nn.Dropout(self.dout_p)
        # self.visual = False
        self.visual = True
        self.attn_holder = nn.Identity()

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask=None):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
        '''
        Bq, Sq, d_model_Q = Q.shape # fixes for dummy Q
        B, _, _ = V.shape # fixes for dummy Q
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(Bq, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        if self.visual:
            Q, self.attn_matrix = attention(Q, K, V, mask, self.dropout, self.visual)
            self.attn_matrix = self.attn_matrix.mean(-3)
            _ = self.attn_holder(self.attn_matrix)
        else:
            Q = attention(Q, K, V, mask, self.dropout)
        # (B, Sq, D) <- (B, H, Sq, d_k)
        # Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        # (B, Sq, Dq)
        # Q = self.linear_d2Q(Q)

        return Q






# ==================================================



# EXPERIMENTAL SMART TOKEN POOLING WITH BOTH STATIC AND DYNAMIC TOKENS



# NEW: Fusion module with smart spatial token pooling
class SmartPoolingTransformerEmbModelV2(nn.Module):
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

        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = None
        if 'FC_LAYERS' in cfg.MODEL.EMBEDDER_MODEL:
            fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        
        self.pooling = SmartPoolingV2(cfg)

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
            assert self.smart_final in ['max','one']

        # NEW - backbone warmup: only start fine-tuning the backbone after a certain number of epochs
        self.in_backbone_warmup = False
        if "BACKBONE_WARMUP" in self.cfg.TRAIN:
            self.in_backbone_warmup = True
            print('BACKBONE_WARMUP Enabled, up to epoch ' + str(self.cfg.TRAIN.BACKBONE_WARMUP))



    # TODO EXPERIMENTAL - warmup contol switch
    def set_warmup_status(self, new_status):
        self.in_backbone_warmup = new_status
        print('DEBUG - BACKBONE WARMUP STATUS SET TO: ' + str(self.in_backbone_warmup))
        return



    # NEW - experimental, optionally append cls_emb to other smart tokens, if provided
    def forward(self, x, video_masks=None, cls_emb=None):
        
        # WARM-UP: stop fine-tuning of backbone during warmup period
        if self.in_backbone_warmup:
            x = x.detach()

        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)
        
        # x = self.pooling(x)
        x = self.pooling(x, cls_emb, batch_size) # TODO - NEW EXPERIMENTAL - using cls for dynamic input

        _, c, ntok = x.shape # new c after pooling
        x = x.view(batch_size*num_steps, c, ntok)
        x = torch.movedim(x,2,1)

        # OPTION 1: One-Hot appended directly after smart token pooling
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

        # OPTION 2: One-Hot appended directly before video encoder
        if self.one_hot_pos == "enc":
            one_hot = torch.eye(ntok, device=x.get_device())
            one_hot = torch.unsqueeze(one_hot, 0)
            one_hot = torch.unsqueeze(one_hot, 2)
            one_hot = torch.cat([one_hot]*x.shape[0],0)
            one_hot = torch.cat([one_hot]*x.shape[2],2)
            x = torch.cat([x, one_hot], 3)

        x = x.reshape([batch_size, ntok*num_steps, x.size(3)])

        # TODO - add a SPATIAL POSITION EMBEDDING

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

        x = x.reshape(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x




# NEW: Smart pooling module with cross-attention and learned queries
# for use with SmartPoolingTransformerEmbModel
class SmartPoolingV2(nn.Module):
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
            self.spc = 384 # TODO - need to set this match the CLS dimension
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

        self.cross_att = SmartCrossAttentionV2(cfg=self.cfg, num_static=self.nst, num_dynamic=self.nsdt, 
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
        _, ncc = dyn_in.shape
        dyn_in = dyn_in.view(batch_size, -1, ncc)
        x_out = []
        for i in range(batch_size):
            x_c = x[i,...]
            d_c = dyn_in[i,...]
            x_c = self.cross_att(x_c, x_c, d_c)
            x_c = x_c[:,0,:,:]
            x_c = torch.movedim(x_c, 2, 1)
            x_out.append(x_c)
        x = torch.cat(x_out, 0)

        # old version handled all together
        # x = self.cross_att(x, x, dyn_in)
        # x = x[:,0,:,:]
        # x = torch.movedim(x, 2, 1)
        
        return x



'''
NEW - modified version of MultiheadedAttention for smart token pooling
for use with SmartPooling
NEW - replaces the query input with two query paths:
1) static queries - fixed for all inputs
2) dynamic queries - predicted base on some high-level input vector

Changes:
    Fixed H to be 1 (single headed attention) - TODO
    Removed d_model_Q, as it is not relevant - TODO
    Removed self.linear_Q2d, as it is not relevant - TODO
    Removed self.d_out as it is not relevant
    Removed self.d_k as it is not relevant
    Added controls for the number of static and dynamic smart tokens - TODO
    Made d_model mandatory, as it now specifies the hidden size
    Replaced the 
    Added direct pass through mode, where the input features are treated directly as the values
    Added d_dyn_in - the size of the input for the dynamic token control feature(s)
'''
class SmartCrossAttentionV2(nn.Module):
    def __init__(self, cfg, num_static, num_dynamic, d_model_K, d_model_V, d_model, d_dyn_in=None, dout_p=0.0):
        super(SmartCrossAttentionV2, self).__init__()
        self.cfg = cfg
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.d_model = d_model
        
        # NEW - VALPASS - direct pass through of the base feature vector as the value vector
        self.pass_through = False
        if "VAL_PASS" in self.cfg.MODEL.EMBEDDER_MODEL:
            self.pass_through = self.cfg.MODEL.EMBEDDER_MODEL.VAL_PASS
            # TODO DEBUG
            print('VAL_PASS ENABLED')

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
            
            # init option 1:
            # nn.init.kaiming_uniform_(self.Q_s, a=math.sqrt(5)) # same as nn.Linear weight
            # self.dummy_bias = None
            # print('INIT OPTION: A')

            # init option 2: (same as 1 with dummy bias layer to emulate nn.Linear init)
            nn.init.kaiming_uniform_(self.Q_s, a=math.sqrt(5)) # same as nn.Linear weight
            self.dummy_bias = nn.Parameter(torch.empty(self.d_model, dtype=torch.float32))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Q_s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.dummy_bias, -bound, bound)
            # print('INIT OPTION: B')

            # init option 3:
            # nn.init.xavier_uniform_(self.Q_s)
            # print('INIT OPTION: X')

        # dynamic smart token query generation
        self.num_d = num_dynamic
        self.dyn = False
        if self.num_d > 0:
            self.dyn = True
            self.d_dyn_in = d_dyn_in
            if d_dyn_in is None:
                print('SmartCrossAttentionV2 - WARNING: d_dyn_in in not set, defaulting to equal d_model_V')
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
        # DEBUG - TODO - EXPERIMENTAL
        if self.dummy_bias is not None:
            Q_temp = self.Q_s + self.dummy_bias
        else:
            Q_temp = self.Q_s
        # Q_temp = self.Q_s

        if not self.stat:
            # only dynamic tokens
            Q = Q_d
            Bq = Bdi
        elif not self.dyn:
            # only static tokens
            Q = Q_temp
            Bq = 1
        else:
            # both static and dynamic tokens
            if self.dyn_ctrl == 'separate':
                Q_s_B = torch.cat([Q_temp]*B, 0)
            else:
                Q_s_B = Q_temp
            # Q = torch.cat([Q_d, Q_s_B], 1)
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
            ret, self.attn_matrix = attention(Q, K, V, mask, self.dropout, self.visual)
            self.attn_matrix = self.attn_matrix.mean(-3)
            _ = self.attn_holder(self.attn_matrix)
        else:
            ret = attention(Q, K, V, mask, self.dropout)

        # DEBUG - check for updates in static query params
        # print(self.Q_s)
        # print(Q_temp)

        return ret













# ==================================================
# TOOLS
# ==================================================



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



# Module size printer with annotation for debugging
# TODO - REMOVE
def note_print(x, i):
    # print(str(i) + ': ' + str(x.size()))
    return