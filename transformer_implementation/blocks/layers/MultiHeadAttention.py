import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implements a multi-head attention module in PyTorch.
    
    This class is a child of the PyTorch nn.Module class. It uses scaled dot 
    product attention mechanism and includes dropout for regularization.

    Attributes
    ----------
    n_head : int
        The number of attention heads.
    n_embd : int
        The size of the input and output feature vectors.
    dropout : float
        The dropout rate to use for regularization.
    block_size : int
        The size of the block to use for the attention mask.
    q_attn : torch.nn.Linear
        The query projection layer.
    k_attn : torch.nn.Linear
        The key projection layer.
    v_attn : torch.nn.Linear
        The value projection layer.
    c_proj : torch.nn.Linear
        The output projection layer.
    attn_dropout : torch.nn.Dropout
        The dropout layer for the attention mechanism.
    resid_dropout : torch.nn.Dropout
        The dropout layer for the output.
    bias : torch.Tensor
        The attention mask to ensure causal attention.

    Methods
    -------
    scaled_dot_product_attention(q, k, v, mask: bool = None):
        Computes the scaled dot product attention.
    forward(q_x, k_x, v_x, mask = None, is_masked = False):
        Computes the forward pass of the multi-head attention.

    Parameters
    ----------
    config : object
        A configuration object with the following attributes:
            n_head (int): The number of attention heads.
            n_embd (int): The size of the input and output feature vectors.
            bias (bool): If True, the linear layers will include a bias term.
            dropout (float): The dropout rate to use for regularization.
            block_size (int): The size of the block to use for the attention mask.
    """

    def __init__(self, config):
        """
        Initializes the multi-head attention network with the given configuration.

        Parameters
        ----------
        config : object
            A configuration object with the following attributes:
                n_head (int): The number of attention heads.
                n_embd (int): The size of the input and output feature vectors.
                bias (bool): If True, the linear layers will include a bias term.
                dropout (float): The dropout rate to use for regularization.
                block_size (int): The size of the block to use for the attention mask.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        # INPUTS: query, key, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # OUTPUT: output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

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

        Parameters
        ----------
        q : torch.Tensor
            The query tensor.
        k : torch.Tensor
            The key tensor.
        v : torch.Tensor
            The value tensor.
        mask : bool, optional
            The attention mask. If None, no mask is applied. Default is None.

        Returns
        -------
        tuple
            The output tensor and the attention weights.
        """
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Step 1 & 2: (MatMul) and (Scale)
            
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1) # Step 3: Softmax
        att_weights = att  # Save attention weights for visualization
        
        if self.training:
            att = self.attn_dropout(att)
            
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) # Step 4: MatMul
        return y, att_weights

    def forward(self, q_x, k_x, v_x, mask = None, is_masked = False):
        """
        Implements the forward pass of the multi-head attention.

        Parameters
        ----------
        q_x : torch.Tensor
            The input query tensor.
        k_x : torch.Tensor
            The input key tensor.
        v_x : torch.Tensor
            The input value tensor.
        mask : bool, optional
            The attention mask. If None, no mask is applied. Default is None.
        is_masked : bool, optional
            Define if this MHA is a Masked MHA. Do we have to add or not a triangular mask ?

        Returns
        -------
        tuple
            The output tensor and the attention weights.
        """
        B_q, T_q, C_q = q_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B_kv, T_kv, C_kv = k_x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.q_attn(q_x), self.k_attn(k_x), self.v_attn(v_x)
        q = q.view(B_q, T_q, self.n_head, C_q // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B_kv, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B_kv, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # In case of masked attention layer (decoder)
        if is_masked and mask is None:
            mask = self.bias[:,:,:T_q,:T_q].to(q.device)
            
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        y, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C_q) # Step 5: Concatenate
        
        # output projection
        y = self.resid_dropout(self.c_proj(y)) # Step 6 : Linear
        
        return y, attn_weights