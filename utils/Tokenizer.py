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
            int: The size of the vocabulary.
        """
        return self.encoder.n_vocab


    def sequence_padding(self, sequence, max_size: int = 512, device: str = "cpu") -> torch.Tensor:
        """
        Method to add BOS/PAD/EOS special tokens and ensure the sequence length is within the maximum size.

        Args:
            sequence (torch.Tensor or list): The input sequence.
            max_size (int, optional): The maximum allowed size for the sequence. Defaults to 512.
            device (str, optional): The device where the tensors will be allocated. Defaults to "cpu".

        Returns:
            torch.Tensor: The processed sequence with special tokens added and length limited.
        """
        assert max_size > 2, f"[max_size]: {max_size} should be greater than 2"
        # Ensure the sequence is a torch tensor
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).to(device) if not torch.is_tensor(sequence) else sequence.to(device)

        # Limit the sequence length within (max_size - 2) where 2 corresponding to bos and eos tags
        cutted_sequence_size = max(0, min(max_size - 2, tensor_sequence.size()[0]))
        tensor_sequence = tensor_sequence[:cutted_sequence_size]
        
        # Add BOS token
        tensor_sequence = torch.cat([torch.tensor([self.BOS_IDX], dtype=torch.long, device=device), tensor_sequence], dim=0)

        # Calculate the padding size
        padding_size = max_size - tensor_sequence.size()[0] - 1 # expected size - current size - EOS tag

        # Create PAD tensor
        pad_tensor = torch.full((padding_size,), self.PAD_IDX, dtype=torch.long, device=device)

        # Add PAD and EOS tokens
        tensor_sequence = torch.cat([tensor_sequence, pad_tensor, torch.tensor([self.EOS_IDX], dtype=torch.long, device=device)], dim=0)
        
        return tensor_sequence
    
    def sequence_cleaner(self, sequence):
        """ Method used to remove BOS/PAD/EOS special tokens """
        # Checking tensor format
        list_sequence = sequence.tolist() if torch.is_tensor(sequence) else sequence
        def check_special(number):
            return number not in [self.BOS_IDX, self.EOS_IDX, self.PAD_IDX]
        return list(filter(check_special, list_sequence))

    def tokenize(self, sequence, device="cpu") -> list:
        """
        Method to generate a str list of separated tokens token.

        Args:
            sequence (torch.Tensor or list): The input sequence.
            device (str, optional): The device where the tensors will be allocated. Defaults to "cpu".

        Returns:
            list: The processed sequence converted in a list of tokens in string format.
        """
        # Ensure the sequence is a torch tensor
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).to(device) if not torch.is_tensor(sequence) else sequence.to(device)
        # create batch of idx tokens
        tensor_sequence = tensor_sequence.unsqueeze(0).T
        # decode all batch to recreate list of separated tokens 
        tensor_sequence = self.encoder.decode_batch(tensor_sequence.detach().tolist())
        return tensor_sequence

    def tokenize_from_str(self, sequence, device="cpu") -> list:
        return self.tokenize(self.encoder.encode(sequence), device)