# moved from transformer.py

import math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(Q, K, V, mask=None, dropout=None, visual=False, disjoint=False):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)

    d_k = Q.size(-1)
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)

    # optional: enforce disjoint attention add disjoint attention method
    if disjoint:
        # argmax to one-hot based on: https://discuss.pytorch.org/t/how-to-convert-argmax-result-to-an-one-hot-matrix/125508
        nt = softmax.shape[2]
        pred = torch.argmax(softmax, dim=2)
        dis_mask = F.one_hot(pred, nt)
        dis_mask = torch.permute(dis_mask, [0, 1, 3, 2])
        softmax = softmax * dis_mask
        # softmax = F.softmax(softmax, dim=-1) # post softmax?

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