# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1) # -1 is the last dimension
    
    def forward(self, indices):
        embedded = self.embedding(indices)
        embedded_with_pos = self.positional_encoding(embedded)
        attn_maps = []
        output = embedded_with_pos
        for layer in self.layers:
            output, attn_map = layer(output)
            attn_maps.append(attn_map)
        
        # Predict at each position
        logits = self.fc_out(output)
        log_probs = self.log_softmax(logits)
        
        return log_probs, attn_maps



# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()
        self.query_layer = nn.Linear(d_model, d_internal)  # [d_model, d_internal]
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)
        self.d_internal = d_internal
        self.linear1 = nn.Linear(d_internal, d_internal)
        self.relu = nn.ReLU() # non-linearity
        self.linear2 = nn.Linear(d_internal, d_model)

    def forward(self, input_vecs):
        # [batch_size, seq_len, d_internal]
        queries = self.query_layer(input_vecs)
        keys = self.key_layer(input_vecs)
        values = self.value_layer(input_vecs)

        # Transpose keys matrix [batch_size, d_internal, seq_len]
        keys = keys.transpose(-2, -1)

        # Compute self-attention scores [batch_size, seq_len, seq_len]
        scores = torch.matmul(queries, keys) / math.sqrt(self.d_internal)
        attn_scores = torch.softmax(scores, dim=-1) # softmax the last dimension

        # Apply attention scores to values to output [batch_size, seq_len, d_internal]
        weighted_values = torch.matmul(attn_scores, values)

        # Feedforward layers to output [batch_size, seq_len, d_model]
        ff_out = self.linear2(self.relu(self.linear1(weighted_values)))
        return ff_out, attn_scores


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        # Use LongTensor when you need to store and manipulate integers, particularly for tasks like indexing or when 
        # handling categorical data (e.g., labels in classification tasks). Use the default tensor (floating-point) for
        # most mathematical and neural network operations, which typically involve floating-point computations.
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    model = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=128, num_classes=3, num_layers=1) # 27 characters, 3 classes
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.NLLLoss()

    for epoch in range(10):
        total_loss = 0.0
        for example in train:
            optimizer.zero_grad()
            log_probs, _ = model(example.input_tensor)
            loss = loss_fn(log_probs.squeeze(0), example.output_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
    
    # Evaluate on dev set
    model.eval()
    decode(model, dev_examples=dev)
    return model

####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                # plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))