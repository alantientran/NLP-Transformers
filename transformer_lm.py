import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise NotImplementedError("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise NotImplementedError("Only implemented in subclasses")

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
        indices_to_embed = torch.arange(input_size, device=x.device)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

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

    def create_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = self.create_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def get_next_char_log_probs(self, context):
        self.eval()
        if not context:
            context = ' '  # Use space as start-of-sequence token
        
        context_indices = [self.vocab_index.index_of(c) for c in context[-self.num_positions:]]
        context_tensor = torch.LongTensor(context_indices).unsqueeze(1).to(next(self.parameters()).device)
        with torch.no_grad():
            output = self.forward(context_tensor)
            log_probs = torch.log_softmax(output[-1], dim=-1)
        return log_probs.squeeze().cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.eval()
        log_prob = 0.0
        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(context)
            log_prob += char_log_probs[self.vocab_index.index_of(char)]
            context += char
        return log_prob

def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, num_positions=20, d_model=64, d_internal=256, num_layers=4)
    model.vocab_index = vocab_index

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    chunk_size = 20
    num_epochs = 1  # Increased number of epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_chunks = len(train_text) // chunk_size

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size
            chunk = ' ' + train_text[chunk_start:chunk_end - 1]  # Add space as start-of-sequence token
            next_char = train_text[chunk_end - 1] if chunk_end <= len(train_text) else train_text[0]
            
            input_indices = [vocab_index.index_of(c) for c in chunk]
            target_indices = [vocab_index.index_of(c) for c in chunk[1:] + next_char]

            input_tensor = torch.LongTensor(input_indices).unsqueeze(1)
            target_tensor = torch.LongTensor(target_indices)

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output.squeeze(1), target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / num_chunks}")

        # Evaluate on dev set
        model.eval()
        dev_loss = 0
        num_dev_chunks = len(dev_text) // chunk_size
        with torch.no_grad():
            for i in range(num_dev_chunks):
                dev_chunk = ' ' + dev_text[i*chunk_size:(i+1)*chunk_size - 1]  # Add space as start-of-sequence token
                next_char = dev_text[(i+1)*chunk_size - 1] if (i+1)*chunk_size <= len(dev_text) else dev_text[0]
                
                input_dev = torch.LongTensor([vocab_index.index_of(c) for c in dev_chunk]).unsqueeze(1)
                target_dev = torch.LongTensor([vocab_index.index_of(c) for c in dev_chunk[1:] + next_char])
                output = model(input_dev)
                dev_loss += criterion(output.squeeze(1), target_dev).item()

    return model