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


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers):
        super(NeuralLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions, batched=True)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.num_positions = num_positions

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.positional_encoding(embedded)
        transformer_output = self.transformer_encoder(encoded)
        output = self.fc_out(transformer_output)
        return output

    def get_next_char_log_probs(self, context):
        self.eval()
        if not context:
            return np.log(np.ones(self.vocab_size) / self.vocab_size)
        
        context_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in context[-self.num_positions:]])
        context_tensor = context_tensor.unsqueeze(0)
        log_probs = self(context_tensor)
        return log_probs[0, -1].detach().numpy()

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
    model = NeuralLanguageModel(vocab_size, num_positions=100, d_model=128, d_internal=256, num_layers=4)
    model.vocab_index = vocab_index

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    chunk_size = 100
    batch_size = 32
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
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
                chunk = batch_text[j:j + chunk_size]
                next_char = batch_text[j + chunk_size] if j + chunk_size < len(batch_text) else batch_text[0]

                # Pad the chunk if it's shorter than chunk_size
                if len(chunk) < chunk_size:
                    chunk = chunk + ' ' * (chunk_size - len(chunk))

                input_batch.append([vocab_index.index_of(c) for c in chunk])
                target_batch.append([vocab_index.index_of(c) for c in chunk[1:] + next_char])

            input_batch = torch.LongTensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            optimizer.zero_grad()
            log_probs = model(input_batch)
            loss = loss_fn(log_probs.view(-1, vocab_size), target_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches}")

        # Evaluate on dev set
        model.eval()
        dev_loss = 0
        num_dev_batches = len(dev_text) // chunk_size
        for i in range(num_dev_batches):
            dev_chunk = dev_text[i * chunk_size:(i + 1) * chunk_size]
            next_char = dev_text[(i + 1) * chunk_size] if (i + 1) * chunk_size < len(dev_text) else dev_text[0]

            # Pad the chunk if it's shorter than chunk_size
            if len(dev_chunk) < chunk_size:
                dev_chunk = dev_chunk + ' ' * (chunk_size - len(dev_chunk))

            input_dev = torch.LongTensor([[vocab_index.index_of(c) for c in dev_chunk]])
            target_dev = torch.LongTensor([vocab_index.index_of(c) for c in dev_chunk[1:] + next_char])
            log_probs = model(input_dev)
            dev_loss += loss_fn(log_probs.squeeze(0), target_dev).item()

        print(f"Dev Loss: {dev_loss / num_dev_batches}")

    return model
