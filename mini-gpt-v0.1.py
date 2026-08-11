import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()

        # 1. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 2. Stacking Transformer Decoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = embed_dim,
                nheads = num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first = True
            ) for _ in range(num_layers)
        ])

        # 3. Final normalization and linear projector to vocabulary logits
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, idx):
        batch_size, seq_len = idx.shape

        # Generate position indices [0, 1, ..., seq_len-1]
        pos = torch.arange(0, seq_len, device=idx.device).unsqeeze(0)

        # Combine word meaning with word position
        x = self.token_embedding(idx) + self.position_embedding(pos)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(idx.device)

        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.max_seq_len:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomials(probs, num_samples=1)


