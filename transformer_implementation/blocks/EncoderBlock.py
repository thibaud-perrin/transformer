import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from . import LayerNorm, MultiHeadAttention, FeedForward

class EncoderBlock(nn.Module):
    """
    A class that implements a single encoder block in the Transformer model.
    
    Each block consists of two sub-layers: a multi-head self-attention mechanism,
    and a position-wise fully connected feed-forward network. There is a residual 
    connection around each of the two sub-layers, followed by layer normalization.

    Attributes:
        - ln_1 (LayerNorm): Layer normalization before the multi-head attention layer.
        - attn (MultiHeadAttention): Multi-head attention layer.
        - ln_2 (LayerNorm): Layer normalization before the feed-forward network.
        - ffw (FeedForward): Position-wise feed-forward network.

    Args:
        - config (Config): A configuration object with attribute `n_embd` and `bias`.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffw = FeedForward(config)

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            - x (torch.Tensor): The input tensor to the forward pass.
            - mask (torch.Tensor, optional): The mask tensor to ignore padding, size (B, 1, 1, T).

        Returns:
            - torch.Tensor: The output tensor of the block.
            - decoder_attn: The attention weight of the current block.
        """
        # MultiHeadAttention
        x = self.ln_1(x)
        x_attn, decoder_attn = checkpoint(self.attn, x, x, x, False, mask)
        x = x + x_attn
        # FeedForward
        x = x + checkpoint(self.ffw, self.ln_2(x))
        return x, decoder_attn