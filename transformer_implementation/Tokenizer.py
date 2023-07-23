import tiktoken
import torch
from torch.nn import functional as F

class Tokenizer():
    """
    Implements a Tokenizer based on the tiktoken library for encoding and decoding sequences.
    
    The tokenizer has special tokens for the beginning of sentence (BOS), end of sentence (EOS), and padding (PAD). 
    The sequence can be padded to a fixed length for processing in batch.
    
    Attributes
    ----------
    BOS_IDX : int
        The index of the Beginning of Sentence (BOS) token.
    EOS_IDX : int
        The index of the End of Sentence (EOS) token.
    PAD_IDX : int
        The index of the Padding (PAD) token.
    encoder : tiktoken.Encoding
        The encoding object used for converting sequences to and from tokens.

    Methods
    -------
    vocab_size() -> int:
        Returns the size of the vocabulary.
    sequence_padding(sequence, max_size: int, device: str) -> torch.Tensor:
        Returns the padded sequence as a tensor.
    sequence_cleaner(sequence) -> list:
        Returns the cleaned sequence without any special tokens.
    generate_padding_mask(seq, triu: bool, device: str) -> torch.Tensor:
        Returns a mask for the padding tokens in the sequence.
    tokenize(sequence, device: str) -> list:
        Returns the tokenized sequence.
    tokenize_from_str(sequence, device: str) -> list:
        Returns the tokenized sequence for a given string.
    """

    def __init__(self):
        """
        Initializes the tokenizer.
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
        Returns the size of the vocabulary.
        
        Returns
        -------
        int
            The size of the vocabulary.
        """
        return self.encoder.n_vocab


    def sequence_padding(self, sequence, max_size: int = 512, device: str = "cpu") -> torch.Tensor:
        """
        Pads the sequence to the max_size with the PAD token.
        
        Parameters
        ----------
        sequence : Union[torch.Tensor, list]
            The sequence to be padded.
        max_size : int, optional
            The maximum size of the sequence after padding. Defaults to 512.
        device : str, optional
            The device where the tensor will be allocated. Defaults to "cpu".
            
        Returns
        -------
        torch.Tensor
            The padded sequence.
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
        tensor_sequence = torch.cat([tensor_sequence, torch.tensor([self.EOS_IDX], dtype=torch.long, device=device), pad_tensor], dim=0)
        
        return tensor_sequence
    
    def sequence_cleaner(self, sequence):
        """
        Removes the special tokens from the sequence.
        
        Parameters
        ----------
        sequence : Union[torch.Tensor, list]
            The sequence to be cleaned.
            
        Returns
        -------
        list
            The cleaned sequence.
        """
        # Checking tensor format
        list_sequence = sequence.tolist() if torch.is_tensor(sequence) else sequence
        def check_special(number):
            return number not in [self.BOS_IDX, self.EOS_IDX, self.PAD_IDX]
        return list(filter(check_special, list_sequence))

    def generate_padding_mask(self, seq, triu = False, device="cpu"):
        """
        Generates a mask for the padding tokens in the sequence.
        
        Parameters
        ----------
        seq : torch.Tensor
            The sequence for which the mask will be generated.
        triu : bool, optional
            If True, the mask will be a upper triangular matrix. Defaults to False.
        device : str, optional
            The device where the tensor will be allocated. Defaults to "cpu".
            
        Returns
        -------
        torch.Tensor
            The mask for the sequence.
        """
        # seq shape is (B, T) where B is batch size and T is sequence length
        # padding mask should be of size (B, 1, 1, T), mask should be True for padding tokens and False for others
        mask = (seq != self.PAD_IDX).unsqueeze(0).unsqueeze(0).to(device)
        if triu:
            seq_length = seq.size(-1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
            mask = mask & nopeak_mask
        return mask.to(device)

    def tokenize(self, sequence, device="cpu") -> list:
        """
        Tokenizes the sequence using the encoder.
        
        Parameters
        ----------
        sequence : Union[torch.Tensor, list]
            The sequence to be tokenized.
        device : str, optional
            The device where the tensor will be allocated. Defaults to "cpu".
            
        Returns
        -------
        list
            The tokenized sequence.
        """
        # Ensure the sequence is a torch tensor
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).to(device) if not torch.is_tensor(self.sequence_cleaner(sequence)) else sequence.to(device)
        # create batch of idx tokens
        tensor_sequence = tensor_sequence.unsqueeze(0).T
        # decode all batch to recreate list of separated tokens 
        tensor_sequence = self.encoder.decode_batch(tensor_sequence.detach().tolist())
        return tensor_sequence

    def tokenize_from_str(self, sequence, device="cpu") -> list:
        """
        Tokenizes the string sequence using the encoder.
        
        Parameters
        ----------
        sequence : str
            The string sequence to be tokenized.
        device : str, optional
            The device where the tensor will be allocated. Defaults to "cpu".
            
        Returns
        -------
        list
            The tokenized sequence.
        """
        return self.tokenize(self.encoder.encode(sequence), device)