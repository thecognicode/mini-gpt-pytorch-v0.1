import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()

        # 1. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Stacking Transformer Decoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = embed_dim,
                nheads = num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first = True
            ) for _ in range(num_layers)
        ])