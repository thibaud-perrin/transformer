import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from . import LayerNorm, MultiHeadAttention, FeedForward

class EncoderBlock(nn.Module):
    """
    Implements an encoder block of a Transformer model in PyTorch.
    
    This class is a child of the PyTorch nn.Module class. It consists of 
    two main components: multi-head attention and feed-forward neural network,
    each followed by a layer normalization.

    Attributes
    ----------
    ln_1 : LayerNorm
        Layer normalization before the multi-head attention block.
    attn : MultiHeadAttention
        Multi-head attention block.
    ln_2 : LayerNorm
        Layer normalization before the feed-forward neural network block.
    ffw : FeedForward
        Feed-forward neural network block.
    
    Methods
    -------
    forward(x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        Computes the forward pass of the encoder block.
    
    Parameters
    ----------
    config : object
        A configuration object with the following attribute:
            n_embd (int): The size of the input and output feature vectors.
            bias (bool): If True, the layer normalization layers will include a bias term.
    """
    def __init__(self, config):
        """
        Initializes the encoder block with the given configuration.
        
        Parameters
        ----------
        config : object
            A configuration object with the following attribute:
                n_embd (int): The size of the input and output feature vectors.
                bias (bool): If True, the layer normalization layers will include a bias term.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffw = FeedForward(config)

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Implements the forward pass of the encoder block. 

        First, it applies layer normalization and then applies multi-head attention. 
        The input is then added to the output of the multi-head attention and passed 
        through another layer normalization. Finally, it applies the feed-forward network 
        and adds its output to the input of the feed-forward network.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor with a size of n_embd.
        mask : Optional[torch.Tensor]
            An optional mask tensor to be applied on the attention mechanism.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The output tensor from the encoder block and the attention tensor from 
            the decoder block.
        """
        # MultiHeadAttention
        x = self.ln_1(x)
        x_attn, decoder_attn = checkpoint(self.attn, x, x, x, mask, False)
        
        # FeedForward
        x = self.ln_2(x + x_attn)
        x = x + checkpoint(self.ffw, x)
        
        return x, decoder_attn