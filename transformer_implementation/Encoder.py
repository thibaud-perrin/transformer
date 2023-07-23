import math
import torch
import torch.nn as nn

from . import LayerNorm, EncoderBlock

class Encoder(nn.Module):
    """
    The Encoder class implements a multi-layer transformer encoder.

    This class inherits from the PyTorch nn.Module class and includes 
    token and position embeddings, dropout, multiple encoder blocks, and
    layer normalization.

    Attributes
    ----------
    config : object
        A configuration object with the following attributes:
            vocab_size (int): The size of the vocabulary.
            block_size (int): The maximum sequence length.
            n_embd (int): The dimension of the embeddings.
            dropout (float): The dropout rate.
            n_layer (int): The number of transformer layers.
            bias (bool): If True, the linear layers will include a bias term.
    encoder : torch.nn.ModuleDict
        A dictionary-like module of several layers:
            wte (torch.nn.Embedding): The token embeddings layer.
            wpe (torch.nn.Embedding): The position embeddings layer.
            drop (torch.nn.Dropout): The dropout layer.
            h (torch.nn.ModuleList): The list of transformer layers.
            ln_f (LayerNorm): The final layer normalization.

    Methods
    -------
    get_num_params(non_embedding: bool = True) -> int:
        Returns the total number of parameters.
    _init_weights(module):
        Initializes the weights of the specified module.
    forward(idx, mask=None):
        Computes the forward pass of the encoder.
    
    Parameters
    ----------
    config : object
        A configuration object with the following attributes:
            vocab_size (int): The size of the vocabulary.
            block_size (int): The maximum sequence length.
            n_embd (int): The dimension of the embeddings.
            dropout (float): The dropout rate.
            n_layer (int): The number of transformer layers.
            bias (bool): If True, the linear layers will include a bias term.
    """
    def __init__(self, config):
        """
        Initializes the encoder with the given configuration.

        Parameters
        ----------
        config : object
            A configuration object with the following attributes:
                vocab_size (int): The size of the vocabulary.
                block_size (int): The maximum sequence length.
                n_embd (int): The dimension of the embeddings.
                dropout (float): The dropout rate.
                n_layer (int): The number of transformer layers.
                bias (bool): If True, the linear layers will include a bias term.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # init all weights
        self.apply(self._init_weights)
        
        # apply special scaled init to the residual projections, based on GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # This initialization is used to preventing the variance of the outputs of each layer from exploding or vanishing
                # during the forward pass through the network.
                # Preventing "vanishing/exploding gradients" problem
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Number of Encoder parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True):
        """
        Returns the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Parameters
        ----------
        non_embedding : bool, optional
            If True, subtracts the number of embedding parameters from the total.

        Returns
        -------
        int
            The total number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the model. Proper weight initialization can help speed up the training process and improve model performance.

        Parameters
        ----------
        module : torch.nn.Module
            The module whose weights will be initialized.
        """
        if isinstance(module, nn.Linear):
            # init Linear layers with normal distribution (Gaussian initialization)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # bias initialization if necessary
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # init Embedding layers with normal distribution (Gaussian initialization)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None):
        """
        Implements the forward pass of the encoder.

        Parameters
        ----------
        idx : torch.Tensor
            The input tensor with indices of tokens in the sequence.
        mask : torch.Tensor, optional
            The mask tensor. If provided, it should have the same size as idx.

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            The output tensor after layer normalization and the list of attention
            matrices from each transformer layer.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # pre-encoder block
        tok_emb = self.encoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.encoder.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.encoder.drop(tok_emb + pos_emb) # Addition of input embd + positional encoding

        # encoders block
        encoder_attn_all = []
        for block in self.encoder.h:
            x, encoder_attn = block(x, mask)
            encoder_attn_all.append(encoder_attn)

        return  self.encoder.ln_f(x), encoder_attn_all