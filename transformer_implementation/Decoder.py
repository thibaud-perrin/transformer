import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import LayerNorm, DecoderBlock

class Decoder(nn.Module):
    """
    Implements a decoder module in PyTorch.

    This class is a child of the PyTorch nn.Module class. The decoder uses 
    embeddings for both the vocabulary and the positions. The core of the 
    decoder is a sequence of DecoderBlock modules. The output of the decoder 
    is then processed by a linear layer to produce the final output.
    
    Attributes
    ----------
    config : object
        A configuration object with the necessary attributes for the decoder.
    decoder : torch.nn.ModuleDict
        A dictionary containing the decoder components, including the 
        embeddings, dropout, DecoderBlock sequence, and LayerNorm.
    lm_head : torch.nn.Linear
        A linear layer for producing the final output of the decoder.
    
    Methods
    -------
    get_num_params(non_embedding: bool = True) -> int:
        Returns the number of parameters in the decoder.
    _init_weights(module):
        Initializes the weights of the specified module.
    forward(idx, enc_output=None, src_mask=None, tgt_mask=None):
        Computes the forward pass of the decoder.
    
    Parameters
    ----------
    config : object
        A configuration object with necessary attributes, including 
        vocab_size, n_embd, block_size, dropout, n_layer, and bias.
    """

    def __init__(self, config):
        """
        Initializes the decoder with the given configuration.
        
        Parameters
        ----------
        config : object
            A configuration object with necessary attributes, including 
            vocab_size, n_embd, block_size, dropout, n_layer, and bias.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
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
        
        Parameters
        ----------
        non_embedding : bool, optional
            If True, does not count the parameters of the position embedding
            layer. Default is True.
        
        Returns
        -------
        int
            The number of parameters in the decoder.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the model.
        
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

    def forward(self, idx, enc_output=None, src_mask=None, tgt_mask=None):
        """
        Computes the forward pass of the decoder.
        
        Parameters
        ----------
        idx : torch.Tensor
            The input tensor with token indices.
        enc_output : torch.Tensor, optional
            The output of the encoder. Default is None.
        src_mask : torch.Tensor, optional
            The mask for the source sequence. Default is None.
        tgt_mask : torch.Tensor, optional
            The mask for the target sequence. Default is None.
        
        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]
            A tuple containing the output tensor, a list of attention scores 
            from the decoder blocks, and a list of cross-attention scores.
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
            x, decoder_attn, cross_attn = block(x, enc_output, src_mask, tgt_mask)
            decoder_attn_all.append(decoder_attn)
            cross_attn_all.append(cross_attn)
            
        x = self.decoder.ln_f(x)
        return self.lm_head(x), decoder_attn_all, cross_attn_all