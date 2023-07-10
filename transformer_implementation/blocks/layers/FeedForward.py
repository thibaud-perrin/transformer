import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A position-wise Feed Forward Neural Network (FFNN) class for transformer models.
    
    The class implementing a position-wise FFNN.
    The FFNN consists of two linear transformations with a GELU activation in between, 
    followed by a dropout for regularization.

    Attributes:
        - c_fc (nn.Linear): First fully connected layer.
        - gelu (nn.GELU): GELU activation function layer.
        - c_proj (nn.Linear): Second fully connected layer.
        - dropout (nn.Dropout): Dropout layer for regularization.

    Args:
        - config (Config): A configuration object with attribute `n_embd`, `bias`, and `dropout`.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        """
        Define the computation performed at every call.

        Args:
            - x (torch.Tensor): The input tensor to the forward pass.

        Returns:
            - torch.Tensor: The output of the FFNN.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x