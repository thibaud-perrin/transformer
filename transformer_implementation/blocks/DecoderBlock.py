import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from . import LayerNorm, MultiHeadAttention, FeedForward

class DecoderBlock(nn.Module):
    """
    Implements a decoder block module in PyTorch, as part of a transformer architecture.
    
    This class is a child of the PyTorch nn.Module class. It includes self-attention,
    cross-attention with the encoder's output, and a feed-forward network.
    
    Attributes
    ----------
    ln_1, ln_2, ln_3 : LayerNorm
        Layer normalization layers that normalize the input tensor before the attention
        and feed-forward networks.
    attn1, attn2 : MultiHeadAttention
        Multi-head attention layers. attn1 is for self-attention, while attn2 is for
        cross-attention with the encoder's output.
    ffw : FeedForward
        The feed-forward network layer.
    
    Methods
    -------
    forward(x: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Computes the forward pass of the network.
    
    Parameters
    ----------
    config : object
        A configuration object with the following attribute:
            n_embd (int): The size of the input and output feature vectors.
            bias (bool): If True, the layer normalization will include a bias term.
    """
    def __init__(self, config):
        """
        Initializes the decoder block with the given configuration.
        
        Parameters
        ----------
        config : object
            A configuration object with the following attribute:
                n_embd (int): The size of the input and output feature vectors.
                bias (bool): If True, the layer normalization will include a bias term.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn1 = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn2 = MultiHeadAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffw = FeedForward(config)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None) -> torch.Tensor:
        """
        Implements the forward pass of the decoder block. The method applies self-attention,
        cross-attention with the encoder's output, and a feed-forward network to the input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor from the previous decoder block or the embedding layer.
        encoder_output : torch.Tensor
            The output tensor from the encoder.
        src_mask : Optional[torch.Tensor], default=None
            The source mask tensor. If provided, it will be used in the cross-attention layer.
        tgt_mask : Optional[torch.Tensor], default=None
            The target mask tensor. If provided, it will be used in the self-attention layer.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The output tensor from the decoder block, and the attention matrices from the 
            self-attention and cross-attention layers.
        """
        # Masked MultiHeadAttention
        x = self.ln_1(x)
        x_attn, encoder_attn = checkpoint(self.attn1, x, x, x, tgt_mask, True)
        
        # MultiHeadAttention with q, k from encoder and x from decoder
        x = self.ln_2(x + x_attn)
        x_attn, cross_attn = checkpoint(self.attn2, x, encoder_output, encoder_output, src_mask, False)
        
        # FeedForward
        x = self.ln_3(x + x_attn)
        x = x + checkpoint(self.ffw, x)
        
        return x, encoder_attn, cross_attn