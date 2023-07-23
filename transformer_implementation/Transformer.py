import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import Encoder, Decoder

class Transformer(nn.Module):
    """
    A PyTorch implementation of a Transformer model.
    
    The Transformer model consists of an Encoder and a Decoder. 
    It supports functionalities like forward pass, token generation, optimizer configuration,
    and save/load model state.
    
    Attributes
    ----------
    config : object
        A configuration object with necessary attributes for Transformer model.
    encoder : Encoder
        The encoder part of the Transformer model.
    decoder : Decoder
        The decoder part of the Transformer model.

    Methods
    -------
    forward(src, tgt, src_mask=None, tgt_mask=None):
        Implements the forward pass of the Transformer model and returns the output and loss.
    generate(src, idx, src_mask=None, max_new_tokens=128, temperature=1.0, top_k=None):
        Generates new tokens given a source tensor.
    configure_optimizers(weight_decay, learning_rate, betas, device_type, eps):
        Configures the AdamW optimizer for the Transformer model.
    save_model(path: str):
        Saves the model state to the given file path.
    load_model(path: str):
        Loads the model state from the given file path.

    Parameters
    ----------
    config : object
        A configuration object with necessary parameters for Transformer model. It includes:
            vocab_size (int): The size of vocabulary.
            block_size (int): The size of a block for Transformer.
            PAD_IDX (int): The index representing padding in token sequence.
    """
    def __init__(self, config):
        """
        Initializes the Transformer model with the given configuration.
        
        Parameters
        ----------
        config : object
            A configuration object with necessary parameters for Transformer model. It includes:
                vocab_size (int): The size of vocabulary.
                block_size (int): The size of a block for Transformer.
                PAD_IDX (int): The index representing padding in token sequence.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        
        # report number of parameters
        print("Total number of parameters: %.2fM" % (self.encoder.get_num_params()/1e6 + self.decoder.get_num_params()/1e6,))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Implements the forward pass of the Transformer model.
        
        Parameters
        ----------
        src : torch.Tensor
            The input tensor for the source sequence.
        tgt : torch.Tensor
            The input tensor for the target sequence.
        src_mask : torch.Tensor, optional
            The input tensor for source sequence masking.
        tgt_mask : torch.Tensor, optional
            The input tensor for target sequence masking.
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            The output tensor post-processed by the Transformer model and the calculated loss.
        """
        assert src.dim() == 2 and tgt.dim() == 2, "src and tgt should be 2D (B, S)"
        if src_mask is not None:
            assert src_mask.dim() == 4, "src_mask should be 4D (B, 1, 1, S)"
        if tgt_mask is not None:
            assert tgt_mask.dim() == 4, "tgt_mask should be 4D (B, 1, 1 S)"

        enc_output, _ = self.encoder(src, src_mask)
        
        # Shift targets for decoder input and add padding at the end
        tgt_shifted = F.pad(tgt[:, :-1], (0, 1), value=self.config.PAD_IDX)
        # Shift targets for loss calculation and add padding at the end
        tgt_shifted_for_loss = F.pad(tgt[:, 1:], (0, 1), value=self.config.PAD_IDX)
        
        output, _, _ = self.decoder(tgt_shifted, enc_output, src_mask, tgt_mask)

        B, T, C = output.shape
        output = output.view(B*T, C)
        tgt_tgt = tgt_shifted_for_loss.view(B*T)
        # Calculate the loss, using both the output and the target
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.PAD_IDX) # Ignore padding tokens
        loss = loss_fct(output, tgt_tgt)
        return output, loss

    @torch.no_grad()
    def generate(self, src, idx, src_mask=None, max_new_tokens=128, temperature=1.0, top_k=None):
        """
        Generates new tokens given a source tensor.
        
        Parameters
        ----------
        src : torch.Tensor
            The input tensor for the source sequence.
        idx : torch.Tensor
            The input tensor with indices in the current context.
        src_mask : torch.Tensor, optional
            The input tensor for source sequence masking.
        max_new_tokens : int, optional
            The maximum number of new tokens to be generated.
        temperature : float, optional
            The softmax temperature for controlling the randomness of predictions.
        top_k : int, optional
            The number of highest probability vocabulary tokens to keep for next step prediction.
        
        Returns
        -------
        torch.Tensor, dict
            The tensor with new generated token indices and a dictionary with attentions.
        """
        enc_output, encoder_attn = self.encoder(src, src_mask)
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # get the predictions
            logits, dec_attention, cross_attention = self.decoder(idx_cond, enc_output, src_mask)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx, dict(encoder_attn=encoder_attn, decoder_attn=dec_attention, cross_attn=cross_attention)
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, eps):
        """
        Configures the AdamW optimizer for the Transformer model.
        
        Parameters
        ----------
        weight_decay : float
            The L2 penalty (regularization) coefficient.
        learning_rate : float
            The learning rate for AdamW optimizer.
        betas : tuple(float, float)
            Coefficients used for computing running averages of gradient and its square.
        device_type : str
            The device type for the optimizer, either "cpu" or "cuda".
        eps : float
            A term added to the denominator to improve numerical stability.
        
        Returns
        -------
        torch.optim.AdamW
            The AdamW optimizer configured for the Transformer model.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)

        return optimizer

    def save_model(self, path: str):
        """
        Saves the model state to the given file path.
        
        Parameters
        ----------
        path : str
            The file path where the model state is to be saved.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model state from the given file path.
        
        Parameters
        ----------
        path : str
            The file path from where the model state is to be loaded.
        """
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist.")
        self.load_state_dict(torch.load(path))