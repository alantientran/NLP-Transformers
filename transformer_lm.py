import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.arange(input_size)).type(torch.LongTensor)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, nhead=8):
        super(NeuralLanguageModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, num_positions, batched=False)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.num_positions = num_positions

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def get_next_char_log_probs(self, context):
        self.eval()
        if not context:
            context = ' '  # Use space as start-of-sequence token
        
        context_indices = [self.vocab_index.index_of(c) for c in context[-self.num_positions:]]
        if len(context_indices) < self.num_positions:
            context_indices = [self.vocab_index.index_of(' ')] + context_indices  # Prepend start-of-sequence token
        
        context_tensor = torch.LongTensor(context_indices).unsqueeze(1)  # Add batch dimension
        with torch.no_grad():
            output = self.forward(context_tensor)
            log_probs = torch.log_softmax(output[-1], dim=-1)
        return log_probs.squeeze().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.eval()
        log_prob = 0.0
        if not context:
            context = ' '  # Use space as start-of-sequence token
        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(context)
            log_prob += char_log_probs[self.vocab_index.index_of(char)]
            context += char
        return log_prob

def train_lm(args, train_text, dev_text, vocab_index, batch_size=32):
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, num_positions=64, d_model=128, d_internal=256, num_layers=4)
    model.vocab_index = vocab_index

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    chunk_size = 20  # Sequence length for each training sample
    num_epochs = 10

    def prepare_batch(text, vocab_index, chunk_size, batch_size):
        """Prepare input and target tensors for a batch of sequences."""
        num_chunks = len(text) // chunk_size
        input_batch = []
        target_batch = []

        for i in range(0, num_chunks, batch_size):
            batch_input = []
            batch_target = []
            for b in range(batch_size):
                if i + b < num_chunks:
                    chunk_start = (i + b) * chunk_size
                    chunk_end = (i + b + 1) * chunk_size
                    chunk = text[chunk_start:chunk_end]
                    next_char = text[chunk_end] if chunk_end < len(text) else text[0]

                    # Prepend space as start-of-sequence token
                    input_indices = [vocab_index.index_of(' ')] + [vocab_index.index_of(c) for c in chunk]
                    target_indices = [vocab_index.index_of(c) for c in chunk + next_char]

                    batch_input.append(input_indices)
                    batch_target.append(target_indices)

            # Padding the sequences in the batch to the same length
            max_len = max(len(seq) for seq in batch_input)
            padded_input = [seq + [vocab_index.index_of(' ')] * (max_len - len(seq)) for seq in batch_input]
            padded_target = [seq + [vocab_index.index_of(' ')] * (max_len - len(seq)) for seq in batch_target]

            input_batch.append(torch.LongTensor(padded_input))
            target_batch.append(torch.LongTensor(padded_target))

        return input_batch, target_batch

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        input_batches, target_batches = prepare_batch(train_text, vocab_index, chunk_size, batch_size)

        for input_batch, target_batch in zip(input_batches, target_batches):
            optimizer.zero_grad()

            # Input batch shape: [batch_size, sequence_length]
            # Transpose it to [sequence_length, batch_size] for the transformer
            input_tensor = input_batch.transpose(0, 1)
            target_tensor = target_batch.transpose(0, 1)

            output = model(input_tensor)
            
            # Using .reshape() to handle non-contiguous memory
            loss = criterion(output.reshape(-1, vocab_size), target_tensor.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(input_batches)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # Evaluation on dev set
        model.eval()
        dev_loss = 0
        dev_input_batches, dev_target_batches = prepare_batch(dev_text, vocab_index, chunk_size, batch_size)
        with torch.no_grad():
            for dev_input_batch, dev_target_batch in zip(dev_input_batches, dev_target_batches):
                input_tensor = dev_input_batch.transpose(0, 1)
                target_tensor = dev_target_batch.transpose(0, 1)

                output = model(input_tensor)
                
                # Using .reshape() for non-contiguous tensors in evaluation too
                dev_loss += criterion(output.reshape(-1, vocab_size), target_tensor.reshape(-1)).item()

        avg_dev_loss = dev_loss / len(dev_input_batches)
        dev_perplexity = math.exp(avg_dev_loss)
        print(f"Dev Loss: {avg_dev_loss}, Dev Perplexity: {dev_perplexity}")

    return model