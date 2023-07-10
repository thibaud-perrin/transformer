# Data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Args:
        - dataset (Dataset): a dataset from HuggingFace datasets library.
        - tokenizer (Tokenizer): The custom tiktoken tokenizer used to encode sequences.
        - block_size (int): The maximum sequence length for tokenization.
    """

    def __init__(self, dataset, tokenizer, block_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __getitem__(self, index):
        """
        Get a tokenized example from the dataset at the specified index.

        Args:
            - index (int): the index of the example to fetch.

        Returns:
            - Dict: dictionary with keys 'inputs', 'targets' and 'translation', containing tokenized input,
            target sequences and original translation.
        """
        translation = self.dataset[index]['translation']
        encode = self.tokenizer.encoder.encode
        inputs = self.tokenizer.sequence_padding(encode(translation['en']), self.block_size) # source language
        targets = self.tokenizer.sequence_padding(encode(translation['fr']), self.block_size) # target language
        return {'inputs': inputs, 'targets': targets, 'translation': translation}

    def __len__(self) -> int :
        """
        Returns the length of the dataset.

        Returns:
            - int: the length of the dataset.
        """
        return self.dataset.num_rows


class DataLoaderFactory():
    """
    A class to instantiate PyTorch DataLoaders for different splits of a HuggingFace Dataset.

    Args:
        - block_size (int): The maximum sequence length for tokenization.
        - batch_size (int): The batch size for DataLoader.
        - tokenizer (Tokenizer): a tokenizer that has an encode method.
        - device (str): 'cpu' or 'cuda', depending on whether we use CPU or GPU.
    """

    def __init__(self, block_size, batch_size, tokenizer, device, train_dataset_size = 5_000_000):
        self.train_data = TranslationDataset(load_dataset("wmt14", "fr-en", split=f"train[:{train_dataset_size}]"), tokenizer, block_size)
        self.val_data = TranslationDataset(load_dataset("wmt14", "fr-en", split="validation"), tokenizer, block_size)
        self.test_data = TranslationDataset(load_dataset("wmt14", "fr-en", split="test"), tokenizer, block_size)

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.dataloader_train = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.dataloader_val = DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
        self.dataloader_test = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

    
    
    def __len__(self) -> int :
        """
        Print the length of each dataset and returns the length of all datasets.

        Returns:
            - int: the length of all dataset (train + val + test).
        """
        print("\033[95m\033[1m\033[4mNumber of data by datasets splits\033[0m")
        print(f"Train\t\t: {len(self.train_data)}\t-> {len(self.train_data)/self.batch_size}")
        print(f"Validation\t: {len(self.val_data)}\t\t-> {len(self.val_data)/self.batch_size}")
        print(f"Test\t\t: {len(self.test_data)}\t\t-> {len(self.test_data)/self.batch_size}")
        total = len(self.train_data) + len(self.val_data) + len(self.test_data)
        print(f"Total\t\t: {total}")
        return total

    def get_batch(self, split):
        """
        Choose the correct DataLoader and yield batches from it.

        Args:
            - split (str): 'train', 'val' or 'test'.

        Yields:
            - Dict: a dictionary with keys 'inputs', 'targets' and 'translation', containing a batch of tokenized input,
            target sequences and original translation.
        """
        # choose the correct dataloader
        if split == 'train':
            dataloader = self.dataloader_train
        elif split == 'val':
            dataloader = self.dataloader_val
        else:
            dataloader = self.dataloader_test

        for batch in dataloader:
            # Separate the 'translation' from the rest of the batch
            translation = batch.pop('translation')
    
            # Move tensors to device
            batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
    
            # Add 'translation' back into the batch
            batch_on_device['translation'] = translation
    
            yield batch_on_device