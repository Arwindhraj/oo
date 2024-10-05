import torch
import torch.nn as nn
from torch.nn import Transformer

class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer = Transformer(d_model=embed_size, nhead=nhead, num_encoder_layers=0,
                                       num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, max_len)

        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float))
        embeddings = self.pos_encoder(embeddings)
        embeddings = embeddings.permute(1, 0, 2)  # (max_len, batch_size, embed_size)

        features = features.unsqueeze(0)  # (1, batch_size, embed_size)

        tgt_mask = self.transformer.generate_square_subsequent_mask(len(captions[0])).to(captions.device)

        output = self.transformer(tgt=embeddings, memory=features, tgt_mask=tgt_mask)
        output = self.fc_out(output)  # (max_len, batch_size, vocab_size)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (max_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)