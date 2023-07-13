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
        tgt_shifted = tgt[:, :-1] # Shifted target
        output, _, _ = self.decoder(tgt_shifted, enc_output, src_mask, tgt_mask[:, :, :-1, :-1])

        # Calculate the loss, using both the output and the target
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.tokenizer.PAD_IDX) # Ignore padding tokens
        # The targets for the loss function are the input sequences shifted
        tgt_tgt = tgt[:, 1:].contiguous()
        loss = loss_fct(output.view(-1, output.size(-1)), tgt_tgt.view(-1))
        return output, loss


    @torch.no_grad()
    def inference(self, src, src_mask=None, tgt_mask=None, max_length=20, top_p=0.95):
        """
        Generates a sequence translation based on the input source sequence using Top P sampling.

        Args:
            - src (torch.Tensor): The input source sequence.
            - src_mask (torch.Tensor): The input_mask tensor to the encoder, size (B, 1, 1, T).
            - tgt_mask (torch.Tensor): The target_masks tensor to the decoder, size (B, 1, 1, T).
            - max_length (int): The maximum length of the generated sequence.
            - top_p (float): The top-p probability threshold for sampling.
            - temperature (float): The temperature value for sampling.

        Returns:
            - str: The generated sequence translation.
        """
        assert src.dim() == 2, "src should be 2D (B, S)"
        if src_mask is not None:
            assert src_mask.dim() == 4, "src_mask should be 4D (B, 1, 1, S)"

        device = self.config.device
        
        enc_output, encoder_attn = self.encoder(src, src_mask)

        # Initialize the target sequence with the start token
        output_seq = [self.config.tokenizer.BOS_IDX]

        # For each possible position in the output sequence...
        for iter in range(max_length):
            # Convert the output sequence list to a tensor
            tgt_tensor = torch.LongTensor(output_seq).unsqueeze(0).to(device)
            
            print(f"\r{iter+1}/{self.config.block_size}", end="")
            
            # Create a target mask
            tgt_mask = self.config.tokenizer.generate_padding_mask(tgt_tensor, True, device)

            # Decode the encoded source sequence
            output, dec_attention, cross_attention = self.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)

            # Take the last token of the output sequence
            output_token = output.argmax(2)[:, -1].item()
            
            # Append the output token to the output sequence
            output_seq.append(output_token)

            # Break the loop if the output token is the end-of-sequence token
            if output_token == self.config.tokenizer.EOS_IDX:
                break
        # Convert the generated target sequence to a string
        return tgt_tensor, dict(encoder_attn=encoder_attn, decoder_attn=dec_attention, cross_attn=cross_attention)
        
    @torch.no_grad()
    def generate(self, src, p=0.9, src_mask=None):
        """
        Generate a sequence of words using Top-p sampling.
    
        Args:
            - src (torch.Tensor): The source sequences to translate.
            - p (float): The cumulative probability threshold for Top-p sampling.
            - src_mask (torch.Tensor): The input_mask tensor to the encoder, size (B, 1, 1, T).
    
        Returns:
            - Tuple[torch.Tensor, Dict[str, torch.Tensor]]: The best sequence found by beam search and a dictionary containing the attention weights.
        """
        enc_output, encoder_attn = self.encoder(src, src_mask)
        # initialize beam with start token
        idx = torch.full((src.size(0), 1), self.config.tokenizer.BOS_IDX).long().to(src.device)

        for iter in range(self.config.block_size):
            print(f"\r{iter+1}/{self.config.block_size} idx={idx}", end="")
            outputs, dec_attention, cross_attention = self.decoder(idx, enc_output)

            # Get the next token probabilities
            next_token_logits = outputs[:, -1, :]
    
            # Sample a token with Top-p sampling
            next_token = self.__top_p_sampling(next_token_logits, p=p)
    
            # Append the sampled token to the idx
            idx = torch.cat([idx, next_token], dim=-1)

        return idx, dict(encoder_attn=encoder_attn, decoder_attn=dec_attention, cross_attn=cross_attention)

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