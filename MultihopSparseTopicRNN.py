# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:28:54 2018

@author: truthless
"""

import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from DataManager import GO
from Dgl_Graph import Graph
from SparseGraphAttention import SGraphAttention

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

g = Graph().get_graph()


class TopicRNN_SGAT(nn.Module):

    def __init__(self, model, rnn_type, word_vector, nvoc, nvoc_nonstop, nembed, nhid, nhid_infer,
                 ntopic, n_entity, nodes_rep, adj, gcn_dropout=0.5, teacher_forcing=0.5, nlayers=1, dropout=0.05):
        super(TopicRNN_SGAT, self).__init__()

        self.edge_number = 0
        self.nhid = nhid  # H
        self.nhid_infer = nhid_infer  # E
        self.ntopic = ntopic  # K
        self.nvoc = nvoc  # C
        self.nvoc_nonstop = nvoc_nonstop  # C-
        self.nembed = nembed  # V
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(nvoc, nembed, nhid, rnn_cell=rnn_type, variable_lengths=True, embedding=word_vector)

        self.fc = nn.Linear(nvoc_nonstop, nhid_infer)
        self.fc_mu = nn.Linear(nhid_infer, nhid_infer)
        self.fc_sigma = nn.Linear(nhid_infer, nhid_infer)
        self.fc_theta = nn.Linear(nhid_infer, ntopic)
        # just use a fully-connected layer to replace gcn
        self.fc_replace_gcn = nn.Linear(nembed, nhid)
        # convert the hidden state to generate a query vector to query at the KG
        self.fc_query = nn.Linear(nembed, nhid)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_decoder = getattr(nn, rnn_type)(nembed, nembed, nlayers, dropout=dropout)
            self.rnn_context = getattr(nn, rnn_type)(nhid, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("rnn_type should be GRU or LSTM! {0}".format(rnn_type))
        self.fc_stop_word = nn.Linear(nembed, 2)
        self.text_decoder = nn.Linear(nembed, nvoc)
        self.topic_decoder = nn.Linear(ntopic, nvoc)

        self.encoder_mem = EncoderMemNN(self.nvoc, self.nembed, 10, 0, 1) #10:hop

        self.n_entity = n_entity
        self.adj = adj.cuda()
        self.nodes_rep = torch.LongTensor(nodes_rep).cuda()
        init = "uniform"

        self.gc1 = SGraphAttention(g, nembed, nhid, 0.02, 5)
        self.gc2 = SGraphAttention(g, nhid, nhid, 0.02, 5)

        self.gcn_dropout = gcn_dropout
        self.gcn_normalize = False

        self.model = model

    def get_nodes_embed(self):
        # note self.nodes_rep[0] corresponds to the node "none"
        nodes_embed = self.encoder.embedding(self.nodes_rep)
        nodes_embed = torch.max(nodes_embed, dim=1)[0]
        # nodes_embed = torch.mean(nodes_embed, dim=1)

        mode = "gcn_no_activation"  # "gcn", "no_gcn", "gcn_no_activation", "gcn_no_dropout", "gcn_no_drop_no_acti"
        mode = self.model

        if mode == "gcn":
            nodes_embed, edges = self.gc1.forawrd(nodes_embed)
            nodes_embed = F.leaky_relu(nodes_embed)
            nodes_embed = F.dropout(nodes_embed, self.gcn_dropout)
            nodes_embed, edges = self.gc2.forawrd(nodes_embed)
            nodes_embed = F.leaky_relu(nodes_embed)
        if mode == "no_gcn":
            nodes_embed = self.fc_replace_gcn(nodes_embed)
        if mode == "gcn_no_activation":
            edges= "__ALL__"
            nodes_embed, edges, edge_numbers1 = self.gc1.forawrd(nodes_embed, edges,skip=0)
            nodes_embed = F.dropout(nodes_embed, self.gcn_dropout)
            nodes_embed, edges, edge_numbers2 = self.gc2.forawrd(nodes_embed, edges,skip=1)
            self.edge_number = edge_numbers2
        if mode == "gcn_no_dropout":
            nodes_embed, edges, edge_numbers = self.gc1.forawrd(nodes_embed)
            nodes_embed = F.leaky_relu(nodes_embed)
            nodes_embed, edges, edge_numbers = self.gc2.forawrd(nodes_embed)
            nodes_embed = F.leaky_relu(nodes_embed)
            self.edge_number += edge_numbers
        if mode == "gcn_no_drop_no_acti":
            nodes_embed, edges = self.gc1.forawrd(nodes_embed)
            nodes_embed, edges = self.gc2.forawrd(nodes_embed)


        return nodes_embed

    def forward(self, x, x_len, y, y_len, x_tf, use_teacher_forcing=None):
        """
        Parameters
        ----------
        x : (batch, sequence length)
        x_len : (batch)
        x_stop : (batch, sequence length)
        x_tf: (batch, nonstop vocabulary size)
        y : (batch, target sequence length)
        y_len : (batch)
        """
        batch_size = x.shape[0]
        hidden = self.encoder_mem(x).unsqueeze(0)
        mu, log_sigma = self.encode(x_tf)  # (batch, E), (batch, E)
        # Compute noisy topic proportions given Gaussian parameters.
        Z = self.reparameterize(mu, log_sigma)  # (batch, E)
        theta = F.softmax(self.fc_theta(Z), dim=1)  # (batch, K)

        target_max_len = max(y_len)
        outputs = torch.zeros_like(y).long().to(device=device)
        word_probs = Variable(torch.zeros(batch_size, target_max_len, self.nvoc)).to(device=device)
        indicator_probs = Variable(torch.zeros(batch_size, target_max_len, 2)).to(device=device)
        entity_probs = Variable(torch.zeros(batch_size, target_max_len, self.n_entity + 1)).to(device=device)

        if use_teacher_forcing is None:
            use_teacher_forcing = random.random() < self.teacher_forcing
        token_input = Variable(torch.LongTensor([GO] * batch_size)).to(device=device)
        rnn_input = self.encoder.embedding(token_input)  # (batch, V)

        for t in range(target_max_len):
            rnn_input = rnn_input.unsqueeze(0)
            rnn_output, hidden = self.rnn_decoder(rnn_input, hidden)
            rnn_output = rnn_output.squeeze(0)  # (batch, H)

            logits = self.text_decoder(rnn_output)  # (batch, C)

            stopword_logits = torch.sigmoid(self.fc_stop_word(rnn_output))  # (batch, 2)
            stopword_predictions = torch.argmax(stopword_logits, dim=-1).unsqueeze(-1)

            topic_additions = self.topic_decoder(theta)  # (batch, C)
            topic_additions[:, :4] = 0  # Padding & Unknowns will be treated as stops.
            topic_mask = (1 - stopword_predictions).expand(-1, self.nvoc)
            topic_additions = topic_additions * topic_mask.float()

            word_logits = logits + topic_additions

            results = torch.argmax(word_logits, dim=-1).detach()
            outputs[:, t] = results

            word_probs[:, t, :] = word_logits
            indicator_probs[:, t, :] = stopword_logits

            if self.model != "TopicRNN":
                # start: entity selection
                nodes_embed = self.get_nodes_embed()  # [314, nhid]
                nodes_embed = nodes_embed.squeeze(1)
                nodes_embed = nodes_embed.squeeze(1)
                zero_node = torch.zeros([1, nodes_embed.shape[1]]).cuda()
                nodes_embed = torch.cat([zero_node, nodes_embed], dim=0)  # [315, nhid]

                # query = self.fc_query(hidden)
                query = hidden
                tmp_hidden = query.squeeze(0).unsqueeze(1).expand(-1, nodes_embed.shape[0],
                                                                  -1)
                nodes_embed = nodes_embed.unsqueeze(0).expand(tmp_hidden.shape)
                entity_prob = torch.sum(tmp_hidden * nodes_embed, dim=2)
                entity_probs[:, t, :] = entity_prob
                # end: entity selection

            if use_teacher_forcing:
                rnn_input = self.encoder.embedding(y[:, t])  # (batch, V)
            else:
                rnn_input = self.encoder.embedding(results)

        return outputs, word_probs, indicator_probs, mu, log_sigma, entity_probs

    def encode(self, x):
        h = F.relu(self.fc(x))  # (batch, E)
        return self.fc_mu(h), self.fc_sigma(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device=device)
        return eps.mul(std).add_(mu)


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


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab  # 577
        self.max_hops = hop  # 1
        self.embedding_dim = embedding_dim  # 64
        self.dropout = dropout  # 0.5
        self.unk_mask = unk_mask  # 1
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, story):
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[
                    0]  # Draw samples from a binomial distribution
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                story = story * a.long().to('cuda')
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0),
                                                          -1).long())  # b * (m * s) * eif (os.cpu_count() > 8)  [batch_size,*,batch_size]
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # query

            prob = self.softmax(torch.sum(m_A * u_temp.to('cuda'), 2))
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)  # output
            u_k = u[-1].to('cuda') + o_k  # query
            u.append(u_k)

        return u_k


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
