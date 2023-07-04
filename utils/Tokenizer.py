import tiktoken
import torch
from torch.nn import functional as F

class Tokenizer():
    """A tokenizer class for encoding/decoding text sequences."""

    def __init__(self):
        """
        Constructor method to initialize special token indices and tokenizer encoding. 
        """
        # Initialize special token indices
        self.BOS_IDX: int = 100264  # Index for the Beginning of Sentence token
        self.EOS_IDX: int = 100265  # Index for the End of Sentence token
        self.PAD_IDX: int = 100266  # Index for the Padding token

        # Initialize base encoding from tiktoken
        cl100k_base = tiktoken.get_encoding("cl100k_base")

        # Initialize the tokenizer's encoding with special tokens added
        self.encoder = tiktoken.Encoding(
            name="cl100k_bep", # Name for the encoder with BOS, EOS, and PAD tokens added
            pat_str=cl100k_base._pat_str, # Pattern string from the base encoding
            mergeable_ranks=cl100k_base._mergeable_ranks, # Mergeable ranks from the base encoding
            special_tokens={
                **cl100k_base._special_tokens, # Special tokens from the base encoding
                "<|bos|>": self.BOS_IDX,  # BOS token
                "<|eos|>": self.EOS_IDX,  # EOS token
                "<|pad|>": self.PAD_IDX,  # PAD token
            }
        )
        
    def vocab_size(self) -> int:
        """
        Method to return the size of the vocabulary in the tokenizer's encoding.

        Returns:
            - int: The size of the vocabulary.
        """
        return self.encoder.n_vocab
        
    def sequence_padding(self, sequence, max_size: int = 512) -> torch.Tensor:
        """
        Method to add BOS/PAD/EOS special tokens and ensure the sequence length is within the maximum size.

        Args:
            - sequence (Union[torch.Tensor, List[int]]): The input sequence.
            - max_size (int, optional): The maximum allowed size for the sequence. Defaults to 512.

        Returns:
            - torch.Tensor: The processed sequence with special tokens added and length limited.
        """
        # Ensure the sequence is a torch tensor
        tensor_sequence = torch.tensor(sequence, dtype=torch.long) if not torch.is_tensor(sequence) else sequence
        
        # Calculate the current sequence length
        sequence_len = tensor_sequence.size()[0]

        # Limit the sequence length within (max_size - 2) where 2 corresponding to bos and eos tags
        cutted_sequence_size = max(0, min(max_size - 2, sequence_len + 2))
        tensor_sequence = tensor_sequence[:cutted_sequence_size]
        
        # Calculate the padding size
        padding_size = max_size - tensor_sequence.size()[0] - 2 # expected size - current size - (BOS tag + EOS tag)
        
        # Add BOS, PAD, and EOS tokens
        tensor_sequence = F.pad(tensor_sequence, (1,0), "constant", self.BOS_IDX)
        tensor_sequence = F.pad(tensor_sequence, (0,padding_size), "constant", self.PAD_IDX)
        tensor_sequence = F.pad(tensor_sequence, (0,1), "constant", self.EOS_IDX)
        
        return tensor_sequence
    
    def sequence_clearner(self, sequence):
        """
        Method to remove BOS/PAD/EOS special tokens from a sequence.

        Args:
            - sequence (torch.Tensor or list): The input sequence.

        Returns:
            - list: The cleaned sequence with special tokens removed.
        """
        # Ensure the sequence is a list
        list_sequence = sequence.tolist() if torch.is_tensor(sequence) else sequence
        
        # Helper function to filter out special tokens
        def check_special(number):
            return number not in [self.BOS_IDX, self.EOS_IDX, self.PAD_IDX]
            
        return list(filter(check_special, list_sequence))