import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A position-wise Feed Forward Neural Network (FFNN) class for transformer models.
    
    The class implementing a position-wise FFNN.
    The FFNN consists of two linear transformations with a GELU activation in between, 
    followed by a dropout for regularization.
    
    Attributes
    ----------
    c_fc : torch.nn.Linear
        The first fully connected layer of the feed-forward network. It takes
        as input a tensor with n_embd features and returns a tensor with 
        4 * n_embd features.
    gelu : torch.nn.GELU
        The Gaussian Error Linear Unit activation function.
    c_proj : torch.nn.Linear
        The second fully connected layer of the feed-forward network. It takes
        as input a tensor with 4 * n_embd features and returns a tensor with 
        n_embd features.
    dropout : torch.nn.Dropout
        The dropout layer for regularization. The dropout rate is specified in
        the configuration.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Computes the forward pass of the network.
    
    Parameters
    ----------
    config : object
        A configuration object with the following attribute:
            n_embd (int): The size of the input and output feature vectors.
            bias (bool): If True, the linear layers will include a bias term.
            dropout (float): The dropout rate to use for regularization.
    """
    def __init__(self, config):
        """
        Initializes the feed-forward network with the given configuration.
        
        Parameters
        ----------
        config : object
            A configuration object with the following attribute:
                n_embd (int): The size of the input and output feature vectors.
                bias (bool): If True, the linear layers will include a bias term.
                dropout (float): The dropout rate to use for regularization.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        """
        Implements the forward pass of the feed-forward network.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor with a size of n_embd.
        
        Returns
        -------
        torch.Tensor
            The output tensor, post-processed by the feed-forward network.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x