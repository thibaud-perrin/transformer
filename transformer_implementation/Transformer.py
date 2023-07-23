import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import Encoder, Decoder

class Transformer(nn.Module):
    """
    This class implements the Transformer model, which includes both the encoder and decoder.

    The Transformer is a sequence transduction model that uses attention mechanisms.
    It is primarily used in tasks that require understanding of context or relationships among words in a text.

    Attributes:
        - encoder (Encoder): The transformer encoder.
        - decoder (Decoder): The transformer decoder.
        - config (:obj:`Config`): The configuration object for the transformer model.

    Args:
        - config (:obj:`Config`): The configuration object with attributes such as `vocab_size`, `block_size`, `n_embd`, `dropout`, `n_layer`, and `bias`.
    """
    def __init__(self, config):
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
        Defines the computation performed at every call.

        Args:
            - src (torch.Tensor): The input tensor to the encoder.
            - tgt (torch.Tensor): The input tensor to the decoder.
            - src_mask (torch.Tensor): The input_mask tensor to the encoder, size (B, 1, 1, T).
            - tgt_mask (torch.Tensor): The target_masks tensor to the decoder, size (B, 1, 1, T).

        Returns:
            - torch.Tensor: The output tensor (logits) of the model.
            - torch.Tensor: The loss tensor calculated on the basis of the decoder's output and target tensor.
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
        Saves the current state of the model to a file.

        Args:
            path (str): The path to the file where the model state should be saved.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model state from a file.

        Args:
            path (str): The path to the file from where the model state should be loaded.

        Raises:
            ValueError: If the specified file does not exist.
        """
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist.")
        self.load_state_dict(torch.load(path))