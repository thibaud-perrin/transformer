import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """A Layer Normalization module with optional bias.

    This implementation of Layer Normalization allows turning off the bias term,
    which is not directly supported by PyTorch's layer normalization function.

    Attributes:
        - weight: Learnable weights for the layer normalization. Initialized as an all ones tensor.
        - bias: Learnable biases for the layer normalization. Initialized as an all zeros tensor if bias argument in constructor is True, otherwise it's set to None.

    Args:
        - ndim: An integer for the dimension of the input vectors.
        - bias: A boolean which, if True, adds a learnable bias to the output.

    """
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """Defines the computation performed at every call.

        Args:
            - input (tensor): The input tensor to be normalized.

        Returns:
            - tensor: The normalized input tensor.

        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)