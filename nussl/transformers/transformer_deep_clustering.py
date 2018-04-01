#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Clustering modeller class
"""

from .. import torch_imported
if torch_imported:
    import torch
    import torch.nn as nn

import numpy as np


class TransformerDeepClustering(nn.Module):
    """
    Transformer Class for deep clustering
    """
    def __init__(self, hidden_size=300, input_size=150, num_layers=2, embedding_size=20):

        if not torch_imported:
            raise ImportError('Cannot import pytorch! Install pytorch to continue.')

        super(TransformerDeepClustering, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,  bidirectional=True,
                      batch_first=True, dropout=0.5)
        linear = nn.Linear(self.hidden_size*2, self.input_size*self.embedding_size)
        self.add_module('rnn', rnn)
        self.add_module('linear', linear)

    def forward(self, input_data):
        """
        Forward training
        Args:
            input_data:

        Returns:

        """
        sequence_length = input_data.size(1)
        num_frequencies = input_data.size(2)
        output, hidden = self.rnn(input_data)
        output = output.contiguous()
        output = output.view(-1, sequence_length, 2*self.hidden_size)
        embedding = self.linear(output)
        embedding = embedding.view(-1, sequence_length*num_frequencies, self.embedding_size)
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    @staticmethod
    def affinity_cost(embedding, assignments):
        """
        Function defining the affinity cost for deep clustering
        Args:
            embedding:
            assignments:

        Returns:

        """
        embedding = embedding.view(-1, embedding.size()[-1])
        assignments = assignments.view(-1, assignments.size()[-1])
        silence_mask = torch.sum(assignments, dim=-1, keepdim=True)
        embedding = silence_mask * embedding
        embedding_transpose = embedding.transpose(1, 0)
        assignments_transpose = assignments.transpose(1, 0)

        class_weights = nn.functional.normalize(torch.sum(assignments, dim=-2),
                                                p=1, dim=-1).unsqueeze(0)
        class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
        weights = torch.mm(assignments, class_weights.transpose(1, 0))
        assignments = assignments * weights.repeat(1, assignments.size()[-1])
        embedding = embedding * weights.repeat(1, embedding.size()[-1])

        loss_est = torch.norm(torch.mm(embedding_transpose, embedding), p=2)
        loss_est_true = 2*torch.norm(torch.mm(embedding_transpose, assignments), p=2)
        loss_true = torch.norm(torch.mm(assignments_transpose, assignments), p=2)
        loss = loss_est - loss_est_true + loss_true
        loss = loss / (loss_est + loss_true)
        return loss

    @staticmethod
    def show_model(model):
        """
        Prints a message to the console with model info
        Args:
            model:

        Returns:

        """
        print(model)
        num_parameters = 0
        for p in model.parameters():
            if p.requires_grad:
                num_parameters += np.cumprod(p.size())[-1]
        print('Number of parameters: {}'.format(num_parameters))
