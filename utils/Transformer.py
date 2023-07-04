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

    # def generate_mask(self, src, tgt):
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    #     seq_length = tgt.size(1)
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     tgt_mask = tgt_mask & nopeak_mask
    #     return src_mask, tgt_mask


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
        # src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        output = self.decoder(tgt, enc_output)

        # Calculate the loss, using both the output and the target
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # The targets for the loss function are the input sequences shifted
        tgt_tgt = tgt[:, 1:].contiguous()
        loss = loss_fct(output.view(-1, output.size(-1)), tgt_tgt.view(-1))
        return output, loss