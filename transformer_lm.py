# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer import Transformer

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers):
        super().__init__()
        self.model = Transformer(vocab_size, num_positions, d_model, d_internal, vocab_size, num_layers)
        self.vocab_size = vocab_size
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        self.model.eval()
        if not context:
            # If context is empty, return uniform distribution
            return np.log(np.ones(self.vocab_size) / self.vocab_size)
        
        context_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in context[-self.num_positions:]])
        context_tensor = context_tensor.unsqueeze(0)
        log_probs, _ = self.model(context_tensor)
        return log_probs[0, -1].detach().numpy()
    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        log_prob = 0.0
        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(context)
            log_prob += char_log_probs[self.vocab_index.index_of(char)]
            context += char
        return log_prob



def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, num_positions=100, d_model=128, d_internal=256, num_layers=4)
    model.vocab_index = vocab_index

    optimizer = optim.Adam(model.model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    chunk_size = 100
    batch_size = 32
    num_epochs = 10

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0

        # Create batches
        num_batches = len(train_text) // (chunk_size * batch_size)
        for i in range(num_batches):
            batch_start = i * chunk_size * batch_size
            batch_end = (i + 1) * chunk_size * batch_size
            batch_text = train_text[batch_start:batch_end]

            input_batch = []
            target_batch = []

            for j in range(0, len(batch_text), chunk_size):
                chunk = batch_text[j:j+chunk_size]
                next_char = batch_text[j+chunk_size] if j+chunk_size < len(batch_text) else batch_text[0]
                
                # Pad the chunk if it's shorter than chunk_size
                if len(chunk) < chunk_size:
                    chunk = chunk + ' ' * (chunk_size - len(chunk))
                
                input_batch.append([vocab_index.index_of(c) for c in chunk])
                target_batch.append([vocab_index.index_of(c) for c in chunk[1:] + next_char])

            input_batch = torch.LongTensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            optimizer.zero_grad()
            log_probs, _ = model.model(input_batch)
            loss = loss_fn(log_probs.view(-1, vocab_size), target_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / num_batches}")

        # Evaluate on dev set
        model.model.eval()
        dev_loss = 0
        num_dev_batches = len(dev_text) // chunk_size
        for i in range(num_dev_batches):
            dev_chunk = dev_text[i*chunk_size:(i+1)*chunk_size]
            next_char = dev_text[(i+1)*chunk_size] if (i+1)*chunk_size < len(dev_text) else dev_text[0]
            
            # Pad the chunk if it's shorter than chunk_size
            if len(dev_chunk) < chunk_size:
                dev_chunk = dev_chunk + ' ' * (chunk_size - len(dev_chunk))
            
            input_dev = torch.LongTensor([[vocab_index.index_of(c) for c in dev_chunk]])
            target_dev = torch.LongTensor([vocab_index.index_of(c) for c in dev_chunk[1:] + next_char])
            log_probs, _ = model.model(input_dev)
            dev_loss += loss_fn(log_probs.squeeze(0), target_dev).item()
        
        print(f"Dev Loss: {dev_loss / num_dev_batches}")

    return model