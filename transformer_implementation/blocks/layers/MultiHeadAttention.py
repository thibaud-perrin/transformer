import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    This module applies multi-head attention mechanism on the input sequence. This implementation doesn't apply mask over the attention scores.
    
    Attributes:
        - n_head (int): Number of attention heads.
        - n_embd (int): Embedding dimensionality.
        - dropout (float): Dropout rate.
        - q_attn (nn.Linear): Linear layer for the query projection.
        - k_attn (nn.Linear): Linear layer for the key projection.
        - v_attn (nn.Linear): Linear layer for the value projection.
        - c_proj (nn.Linear): Linear layer for the output projection.
        - attn_dropout (nn.Dropout): Dropout layer for the attention scores.
        - resid_dropout (nn.Dropout): Dropout layer for the residual connection.
        - flash (bool): Flag indicating if flash attention is available.
    """
    def __init__(self, config):
        """
        Constructor for the MultiHeadAttention class.
        
        Args:
            - config: The configuration object containing model parameters.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.visualize = config.visualize
        
        # INPUTS: query, key, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # OUTPUT: output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # flash attention make GPU go br but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.block_size, config.block_size)
                ).view(1, 1, config.block_size, config.block_size)
            )

    def scaled_dot_product_attention(self, q, k, v, mask: bool = None):
        """
        Computes the scaled dot product attention.
        
        Args:
            - q (Tensor): Query tensor of shape (batch_size, num_heads, seq_length, emb_dim).
            - k (Tensor): Key tensor of shape (batch_size, num_heads, seq_length, emb_dim).
            - v (Tensor): Value tensor of shape (batch_size, num_heads, seq_length, emb_dim).
            - mask (bool, optional): Flag indicating whether to apply mask on the attention scores.

        Returns:
            - y (Tensor): Output tensor after applying attention.
            - attn_weights (list): Attention weights usefull to visualized how attention work
        """
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Step 1 & 2: (MatMul) and (Scale)
        if mask:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Step 3: Softmax
        att_weights = att  # Save attention weights for visualization
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) # Step 4: MatMul
        return y, att_weights

    def forward(self, q_x, k_x, v_x, mask = None):
        """
        Forward pass for the MultiHeadAttention module.
        
        Args:
            - q_x (Tensor): Input query tensor of shape (batch_size, seq_length, emb_dim).
            - k_x (Tensor): Input key tensor of shape (batch_size, seq_length, emb_dim).
            - v_x (Tensor): Input value tensor of shape (batch_size, seq_length, emb_dim).
            - mask (bool, optional): Flag indicating whether to apply mask on the attention scores.

        Returns:
            - y (Tensor): Output tensor after applying multi-head attention.
            - attn_weights (list): Attention weights usefull to visualized how attention work
        """
        B_q, T_q, C_q = q_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B_kv, T_kv, C_kv = k_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.q_attn(q_x), self.k_attn(k_x), self.v_attn(v_x)
        k = k.view(B_kv, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B_q, T_q, self.n_head, C_q // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B_kv, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and not self.visualize:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=mask == True)
            attn_weights = None
        else:
            # manual implementation of attention
            y, attn_weights = self.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C_q) # re-assemble all head outputs side by side # Step 5: Concatenate
        # output projection
        y = self.resid_dropout(self.c_proj(y)) # Step 6 : Linear
        return y, attn_weights