import torch
import torch.nn as nn

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

    def forward(self, src, tgt):
        """
        Defines the computation performed at every call.

        Args:
            - src (torch.Tensor): The input tensor to the encoder.
            - tgt (torch.Tensor): The input tensor to the decoder.

        Returns:
            - torch.Tensor: The output tensor (logits) of the model.
            - torch.Tensor: The loss tensor calculated on the basis of the decoder's output and target tensor.
        """
        enc_output, _ = self.encoder(src)
        tgt_shifted = tgt[:, :-1] # Shifted target
        output, _, _ = self.decoder(tgt_shifted, enc_output)

        # Calculate the loss, using both the output and the target
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.tokenizer.PAD_IDX) # Ignore padding tokens
        # The targets for the loss function are the input sequences shifted
        tgt_tgt = tgt[:, 1:].contiguous()
        loss = loss_fct(output.view(-1, output.size(-1)), tgt_tgt.view(-1))
        return output, loss
        
    @torch.no_grad()
    def translate_beam_search(self, src, beam_size=5):
        """
        Generates translations of the source sequences using beam search.

        Args:
            src (torch.Tensor): The source sequences to translate.
            beam_size (int, optional): The number of beams to use in beam search. Default is 5.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: The best sequence found by beam search and a dictionary containing the attention weights.
        """
        enc_output, encoder_attn = self.encoder(src)
        # initialize beam with start token
        start_token = torch.full((src.size(0), 1), self.config.tokenizer.BOS_IDX).long().to(src.device)   # initialize target tensor with start token bos
        start_score = torch.zeros((src.size(0), 1)).float().to(src.device)  # initialize target tensor with zeros
        beams = [(start_token, start_score, [], [])]  # initialize beams
        for iter in range(self.config.block_size):
            print(f"\r{iter+1}/{self.config.block_size}", end="")
            new_beams = []
            for beam, score, decoder_attentions, cross_attentions in beams:
                output, dec_attention, cross_attention = self.decoder(beam, enc_output)
                output = F.log_softmax(output[:, -1, :], dim=-1)  # use log_softmax for numerical stability

                top_scores, top_indices = output.topk(beam_size, dim=-1)  # select topk predictions
                
                for i in range(beam_size):
                    next_token = top_indices[:, i].unsqueeze(1)  # get next token
                    next_score = top_scores[:, i].unsqueeze(1)  # get next score
                    new_beam = torch.cat((beam, next_token), dim=-1)  # generate new beam
                    new_score = score + next_score  # calculate new score
                    
                    new_beams.append((new_beam, new_score, dec_attention, cross_attention))  # append new beam to new beams

            # sort all candidates by score
            beams = sorted(new_beams, key=lambda tup: tup[1].sum(), reverse=True)[:beam_size]  # keep top performing beams
        return beams[0][0], dict(encoder_attn=encoder_attn, decoder_attn=beams[0][2], cross_attn=beams[0][3])  # return the best sequence


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