import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import LayerNorm, DecoderBlock

class Decoder(nn.Module):
    """
    This class implements the decoder part of the Transformer model.

    The Decoder consists of several DecoderBlocks arranged in sequence. The input first goes through an embedding 
    layer followed by a positional encoding layer. The output of this is then passed through each DecoderBlock in 
    sequence.

    Attributes:
        - decoder (nn.ModuleDict): A dictionary of modules making up the transformer decoder.
        - lm_head (nn.Linear): The final linear layer mapping from the embedding dimension to the vocabulary size.
        - config (:obj:`Config`): The configuration object for the transformer model.

    .. note:: The weight of the embedding layer and the linear layer are shared.

    Args:
        - config (:obj:`Config`): The configuration object with attributes such as `vocab_size`, `block_size`, `n_embd`, `dropout`, `n_layer`, and `bias`.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Learned positional encoding:
            # In this case, instead of using a fixed function to determine positional encoding,
            # we initialize a tensor of positional encodings which gets updated during training via backpropagation.
            # This method may potentially capture more complex position-related patterns than fixed positional encoding,
            # but it also introduces additional parameters to the model.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.decoder.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of Decoder parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Returns the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Args:
            - non_embedding (bool): If True, excludes the position embeddings count from the total. Default is True.

        Returns:
            - int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the model.

        Args:
            - module (torch.nn.Module): The module of the model to be initialized.
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

    def forward(self, idx, enc_output=None, src_mask=None, tgt_mask=None):
        """
        Defines the computation performed at every call.

        Args:
            - idx (torch.Tensor): The input tensor to the forward pass.
            - enc_output (torch.Tensor): The output tensor from the encoder.
            - src_mask (torch.Tensor, optional): The mask tensor to ignore padding, size (B, 1, 1, T).
            - tgt_mask (torch.Tensor, optional): The mask tensor to ignore padding, size (B, 1, 1, T).

        Returns:
            - torch.Tensor: The output tensor (logits) of the model.
            - list: all layers of decoder attentions weights.
            - list: all layers cross attentions weights.
        """
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        tok_emb = self.decoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.decoder.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.decoder.drop(tok_emb + pos_emb) # Addition of input embd + positional encoding

        cross_attn_all = []
        decoder_attn_all = []
        for block in self.decoder.h:
            x, encoder_attn, cross_attn = block(x, enc_output, src_mask, tgt_mask)
            decoder_attn_all.append(encoder_attn)
            cross_attn_all.append(cross_attn)
            
        x = self.decoder.ln_f(x)
        return self.lm_head(x), decoder_attn_all, cross_attn_all