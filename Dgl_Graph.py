# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 14:51 2021

@author: musegay
"""
import torch.nn as nn
import torch
import dgl
import numpy as np
from DataManager import DataManager
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 rnn_cell='GRU', variable_lengths=False, embedding=None, update_embedding=True):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell == 'LSTM':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'GRU':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of
              the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the
              features in the hidden state h
        """
        if self.variable_lengths:
            sort_index = torch.sort(-input_lengths)[1]
            unsort_index = torch.sort(sort_index)[1]
            input_var = input_var[sort_index]
            input_lengths = input_lengths[sort_index]
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.rnn_cell == nn.LSTM:
            hidden = hidden[0]
        if self.variable_lengths:
            hidden = torch.transpose(hidden, 0, 1)[unsort_index]
            hidden = torch.transpose(hidden, 0, 1)
            output, _ = pad_packed_sequence(output, batch_first=True)
            output = output[unsort_index]
        return output, hidden


class Graph():
    def __init__(self):
        super(Graph, self).__init__()
    def get_graph(self):
        manager = DataManager(1000, './data', False, 300,3)
        adj = manager.adj.cuda()
        nodes_rep = manager.nodes_rep
        nodes_rep = torch.LongTensor(nodes_rep).cuda()
        nvoc = len(manager.word2index)
        nembed = 300
        nhid = 64
        word_vector = manager.vector
        rnn_type = 'GRU'

        encoder = Encoder(nvoc, nembed, nhid, rnn_cell=rnn_type, variable_lengths=True,
                          embedding=word_vector)
        encoder.to('cuda')
        self.nodes_embed = encoder.embedding(nodes_rep)

        self.nodes_embed = torch.max(self.nodes_embed, dim=1)[0]
        nodes_edges = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] > 0:
                    nodes_edges.append(([i, j, adj[i][j]]))
        nodes_edges.append(([313,313,0]))

        src_ids = []
        dst_ids = []
        eweight = []

        for node_edge in nodes_edges:
            src_ids.append(node_edge[0])
            dst_ids.append(node_edge[1])
            eweight.append(node_edge[2])

        src_ids = np.array(src_ids)
        dst_ids = np.array(dst_ids)
        eweight = np.array(eweight)

        sp_mat = coo_matrix((eweight, (src_ids, dst_ids)), shape=(314, 314))
        self.g = dgl.from_scipy(sp_mat.astype(float), eweight_name='a',device='cuda')

        self.g.ndata['ft'] = self.nodes_embed.cuda()

        return self