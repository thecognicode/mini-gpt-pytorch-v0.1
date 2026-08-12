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

            # Crop sequence if it exceeds maximum context length
            idx_cond = idx[:, -self.max_seq_len:]

            # Forward pass to get token logits
            logits = self(idx_cond)

            # Focus only on the prediction for the very last token
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token from probability distribution
            idx_next = torch.multinomials(probs, num_samples=1)

            # Appending predicted token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ** Demo Initialization **

vocab_size = 1000 # Size of the dictionary
embed_dim = 128   # Vector size per token
num_heads = 4     # Number of parallel attention heads
num_layers = 4    # Number of stacked transformer blocks
max_seq_len = 32  # Context window length

model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)

prompt_tokens = torch.tensor([[10, 45, 234, 89]])

generated_sequence = model.generate(prompt_tokens, max_new_tokens=10)

print("Input prompt tokens: ", prompt_tokens.tolist())
print("Full generated output: ", generated_sequence.tolist())