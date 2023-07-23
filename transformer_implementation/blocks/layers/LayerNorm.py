import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    A Layer Normalization module with optional bias.

    This implementation of Layer Normalization allows turning off the bias term,
    which is not directly supported by PyTorch's layer normalization function.
    
    Attributes
    ----------
    weight : torch.nn.Parameter
        A learnable scale factor initialized to one. This has the same shape as 
        the input feature dimension.
    bias : torch.nn.Parameter
        A learnable bias term initialized to zero if bias is True, else None. This 
        has the same shape as the input feature dimension.
    
    Methods
    -------
    forward(input: torch.Tensor) -> torch.Tensor:
        Applies layer normalization to the input tensor.

    Parameters
    ----------
    ndim : int
        The feature dimension size of the input tensor.
    bias : bool
        If True, adds a learnable bias to the output.
    """
    def __init__(self, ndim: int, bias: bool):
        """
        Initializes the LayerNorm module with the given parameters.
        
        Parameters
        ----------
        ndim : int
            The feature dimension size of the input tensor.
        bias : bool
            If True, adds a learnable bias to the output.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Implements the forward pass of the LayerNorm module.
        
        Parameters
        ----------
        input : torch.Tensor
            The input tensor that will be normalized.
        
        Returns
        -------
        torch.Tensor
            The normalized output tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)