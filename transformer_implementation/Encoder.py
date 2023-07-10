import math
import torch
import torch.nn as nn

from . import LayerNorm, EncoderBlock

class Encoder(nn.Module):
    """
    A class that implements the encoder part of the Transformer model.

    The encoder consists of several EncoderBlocks arranged in sequence.
    The input first goes through an embedding layer followed by a positional encoding layer.
    The output of this is then passed through each EncoderBlock in sequence.

    Attributes:
        - encoder (nn.ModuleDict): A dictionary of modules making up the transformer encoder.

    Args:
        - config (Config): A configuration object with attributes such as `vocab_size`, `block_size`, `n_embd`, `dropout`, `n_layer`, and `bias`.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Learned positional encoding:
            # In this case, instead of using a fixed function to determine positional encoding,
            # we initialize a tensor of positional encodings which gets updated during training via backpropagation.
            # This method may potentially capture more complex position-related patterns than fixed positional encoding,
            # but it also introduces additional parameters to the model.
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

        Args:
            -non_embedding (bool, optional): If True, excludes the position embeddings count from the total (Default is True).

        Returns:
            - int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the model. Proper weight initialization can help speed up the training process and improve model performance.

        Args:
            - module (nn.Module): The module of the model to be initialized.
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

    def forward(self, idx):
        """
        Defines the computation performed at every call.

        Args:
            - idx (torch.Tensor): The input tensor to the forward pass.
            - targets (torch.Tensor, optional): The target tensor against which the loss will be calculated.

        Returns:
            - torch.Tensor: The output tensor (logits) of the model.
            - list: all encoder layers attentions weights.
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
            x, encoder_attn = block(x)
            encoder_attn_all.append(encoder_attn)

        return  self.encoder.ln_f(x), encoder_attn_all