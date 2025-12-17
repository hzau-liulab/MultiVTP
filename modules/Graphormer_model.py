import torch
from torch import nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        
    def scaled_dot_product_attention(self, q, k, v, short_path_encoding=None, mask=None):
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype, device=q.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if short_path_encoding is not None:
            scores = scores + short_path_encoding.unsqueeze(1)
        softmax = nn.Softmax(dim=-1)
        attn_score = softmax(scores)
        
        context = torch.matmul(attn_score, v)
        return context, attn_score
    
    def forward(self, query, key, value, short_path_encoding=None, mask=None):
        """
        query/key/value dim: (batch, walk_length, embed_dim)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # multilhead: (batch, seq_len, embed_dim) -> (batch, num_heads, walk_length, head_dim)
        q = self.q_proj(query).contiguous().view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).contiguous().view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).contiguous().view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, walk_length, walk_length)
        
        context, attn_score = self.scaled_dot_product_attention(q, k, v, short_path_encoding, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(context)
        return output, attn_score


class GraphormerBlock(nn.Module):
    def __init__(self, in_dim, num_heads=4, hidden_ration=4, drop_out=0.2):
        super(GraphormerBlock, self).__init__()
        self.mha = MultiheadAttention(in_dim, num_heads)
        self.ff = nn.Sequential(nn.Linear(in_dim, in_dim*hidden_ration), nn.ReLU(), nn.Linear(in_dim*hidden_ration, in_dim))
        self.norm1 = nn.LayerNorm(in_dim, eps=1e-6) 
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, x, short_path_encoding=None):
        """
        x dim: (batch, walk_length, embed_dim)
        short_path_encoding dim: (batch, walk_length, walk_length)
        """
        x = self.norm1(x)
        attn_output, _, = self.mha(x, x, x, short_path_encoding)
        attn_output = x + self.dropout1(attn_output)

        ff_output = self.ff(self.norm2(attn_output))
        ff_output = self.dropout2(ff_output)

        out = attn_output + ff_output
        return out