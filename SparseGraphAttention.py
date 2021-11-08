# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 14:51 2021

@author: musegay
"""
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


sig = nn.Sigmoid()
hardtanh = nn.Hardtanh(0,1)
gamma = -0.1
zeta = 1.1
beta =  0.66
eps = 1e-20
const1 = beta * np.log(-gamma/zeta + eps)

def l0_train(logAlpha, min, max):
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def l0_test(logAlpha, min, max):
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def get_loss2(logAlpha):
    return sig(logAlpha - const1)

writer = SummaryWriter()

class SGraphAttention(nn.Module):
    def __init__(self, g, in_dim, out_dim, alpha, bias_l0, l0=1, min=0):
        super(SGraphAttention, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_l = nn.Parameter(torch.Tensor(size=(1,out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1,out_dim)))
        self.bias_l0 = nn.Parameter(torch.FloatTensor([bias_l0]))

        nn.init.xavier_normal_(self.fc.weight.data, gain = 1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain = 1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain = 1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.num = 0
        self.l0 = l0
        self.loss = 0
        self.dis = []
        self.min = min

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        if self.l0 == 0:
            m = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        else:
            tmp = edges.src['a1'] + edges.dst['a2']
            logits = tmp + self.bias_l0

            if self.training:
                m = l0_train(logits, 0, 1)
            else:
                m = l0_test(logits, 0, 1)
            self.loss = get_loss2(logits).sum()
        return {'a': m}

    def norm(self, edges):
        # normalize attention
        a = edges.data['a'] / edges.dst['z']
        return {'a' : a}

    def loop(self, edges):
        # set attention to itself as 1
        return {'a': torch.pow(edges.data['a'], 0)}

    def normalize(self, logits):
        self._logits_name = "_logits"
        self._normalizer_name = "_norm"
        self.g.edata[self._logits_name] = logits
        'copy_edge: computes message using edge feature.'
        self.g.update_all(fn.copy_edge(self._logits_name, self._logits_name),
                         fn.sum(self._logits_name, self._normalizer_name))
        'ndata:Return the data view of all the nodes.'
        return self.g.edata.pop(self._logits_name), self.g.ndata.pop(self._normalizer_name)

    def edge_softmax(self):

        if self.l0 == 0:
            scores = self.softmax(self.g, self.g.edata.pop('a'))

        else:
            scores, normalizer = self.normalize(self.g.edata.pop('a'))

            self.g.ndata['z'] = normalizer.unsqueeze(1)

        self.g.edata['a'] = scores.unsqueeze(1)

    def forawrd(self, inputs, edges='__ALL__', skip=0):
        self.loss =  0
        #prepare
        h = inputs
        ft = self.fc(h)
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1)
        self.g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})

        if skip == 0:
            """Edge feature update by concatenating the features of the destination 
               and source nodes."""
            self.g.apply_edges(self.edge_attention, edges)
            if self.l0 == 1:
                ind = self.g.nodes()
                # loop：set attention to itself as 1
                """Edge feature update by concatenating the features of the destination
                        and source nodes."""
                self.g.apply_edges(self.loop, edges=(ind, ind))

            self.edge_softmax()

            # normal: normalize attention
            if self.l0 == 1:
                """Edge feature update by concatenating the features of the destination
                        and source nodes."""
                self.g.apply_edges(self.norm)

        #2. compute the aggregated node features scaled by the dropped
            edges = self.g.edata['a'].squeeze().nonzero().squeeze()

        self.num = (self.g.edata['a'] > 0).sum()
        self.g.update_all(fn.src_mul_edge('ft','a','ft',), fn.sum('ft','ft'))
        ret = self.g.ndata['ft']

        return ret, edges, self.num
