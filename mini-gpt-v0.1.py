import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()