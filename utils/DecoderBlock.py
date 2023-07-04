import torch
import torch.nn as nn

from . import LayerNorm, MultiHeadAttention, FeedForward

class DecoderBlock(nn.Module):
    """
    A class that implements a single decoder block in the Transformer model.

    Each block consists of three sub-layers: a multi-head self-attention mechanism,
    a multi-head attention mechanism over the encoder's output, and a position-wise 
    fully connected feed-forward network. There is a residual connection around 
    each of the three sub-layers, followed by layer normalization.

    Attributes:
        - ln_1 (LayerNorm): Layer normalization before the first multi-head attention layer.
        - attn1 (MultiHeadAttention): First multi-head attention layer, with self-attention.
        - ln_2 (LayerNorm): Layer normalization before the second multi-head attention layer.
        - attn2 (MultiHeadAttention): Second multi-head attention layer, attends to encoder outputs.
        - ln_3 (LayerNorm): Layer normalization before the feed-forward network.
        - ffw (FeedForward): Position-wise feed-forward network.

    Args:
        config (Config): A configuration object with attribute `n_embd` and `bias`.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn1 = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn2 = MultiHeadAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffw = FeedForward(config)

    def forward(self, x, encoder_output) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            - x (torch.Tensor): The input tensor to the forward pass.
            - encoder_output (torch.Tensor): The output tensor from the last encoder block.

        Returns:
            - torch.Tensor: The output tensor of the block.
        """
        # Masked MultiHeadAttention
        x = self.ln_1(x)
        x = x + self.attn1(x, x, x, True)
        # MultiHeadAttention with q, k from encoder and x from decoder
        x = self.ln_2(x)
        x = x + self.attn2(encoder_output, encoder_output, x)
        # FeedForward
        x = x + self.ffw(self.ln_3(x))
        return x