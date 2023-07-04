from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Data class that stores the configuration for a Transformer model.

    Attributes:
        - tokenizer: An instance of the Tokenizer class.
        - block_size (int): Number of tokens in each sequence. Defaults to 512.
        - batch_size (int): Number of sequences in each batch. Defaults to 12.
        - vocab_size (int): Total size of the tokenizer vocabulary. It is set to the size of the tokenizer vocabulary.
        - n_layer (int): Number of transformer encoder and decoder blocks (N). Defaults to 1.
        - n_head (int): Number of heads in each attention block. Defaults to 2.
        - n_embd (int): Token embedding size. This is from the original Transformer paper. Defaults to 128.
        - dropout (float): Dropout rate to use in the Transformer model. Defaults to 0.1.
        - bias (bool): Indicates whether to use bias in Linears and LayerNorms.
            If True, bias is used similar to GPT-2.
            If False, it is a bit better and faster. Defaults to False.
        - device (str): The device to run the model on. Defaults to 'cpu'. 'cuda' is used if a GPU is available.
        - learning_rate (float): Learning rate for the model optimization. Defaults to 3e-4.
        - eval_interval (int): Number of steps between each validation dataset. Defaults to 1.
        - eval_iters (int): Number of validation epochs. Defaults to 20.
    """
    tokenizer: any
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 1 # 6
    n_head: int = 2 # 8
    n_embd: int = 128 # 512
    dropout: float = 0.1
    bias: bool = False # True:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    learning_rate = 3e-4
    eval_interval = 1
    eval_iters = 20 # 200

    @property
    def vocab_size(self) -> int:
        """Returns the total size of the tokenizer vocabulary.

        Returns:
            - int: The size of the tokenizer vocabulary.
        """
        return self.tokenizer.VocabSize()