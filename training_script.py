import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))

        # Initialize positional encoding tensor
        pe = torch.zeros(max_len, embed_dim)

        # Apply sin to even indices; handle the last index separately if embed_dim is odd
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Exclude the last term for odd embed_dim

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, input_shape, num_heads=19, output_len=1):
        super(MultiHeadAttentionModel, self).__init__()
        n = input_shape[0]
        m = 1 if len(input_shape) == 1 else input_shape[1]

        self.encod = PositionalEncoding(m)
        if m == 1:
            self.attn1 = nn.MultiheadAttention(embed_dim=num_heads * n, num_heads=num_heads, dropout=0.1)
            self.attn2 = nn.MultiheadAttention(embed_dim=num_heads * n, num_heads=num_heads*n, dropout=0.1)
            self.attn3 = nn.MultiheadAttention(embed_dim=num_heads * n, num_heads=num_heads*n*n, dropout=0.1)

        else:
            self.attn1 = nn.MultiheadAttention(embed_dim=m, num_heads=num_heads, dropout=0.1)
            self.attn2 = nn.MultiheadAttention(embed_dim=m, num_heads=num_heads, dropout=0.1)
            self.attn3 = nn.MultiheadAttention(embed_dim=m, num_heads=m, dropout=0.1)

        self.fc2 = nn.Linear(m, output_len)  # Output layer for single number output

    def forward(self, x):
        # Reshape x for multi-head attention
        # x = x.unsqueeze(1)  # Add a sequence length dimension

        x = self.encod_or_fc1(x)

        x, _ = self.attn1(x, x, x)
        x, _ = self.attn2(x, x, x)
        x, _ = self.attn3(x, x, x)

        x = x.mean(dim=1)  # Pooling over the sequence length
        x = self.fc2(x)  # Final output layer
        return x