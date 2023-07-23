# Data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    """
    Dataset for English to French translation tasks. 

    The dataset includes 'translation' field which is a dict that contains 
    source text in English ('en') and target text in French ('fr').

    Attributes
    ----------
    dataset : object
        The dataset object containing translations.
    tokenizer : object
        The tokenizer object used for encoding the translations.
    block_size : int
        The maximum length of the tokenized sequences.

    Methods
    -------
    __getitem__(index: int) -> dict:
        Returns the tokenized input and target sequences, their corresponding masks, 
        and the original translation for a given index.
    __len__() -> int:
        Returns the number of items in the dataset.
    """

    def __init__(self, dataset, tokenizer, block_size):
        """
        Initializes the TranslationDataset with the provided dataset, tokenizer, and block size.

        Parameters
        ----------
        dataset : object
            The dataset object containing translations.
        tokenizer : object
            The tokenizer object used for encoding the translations.
        block_size : int
            The maximum length of the tokenized sequences.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __getitem__(self, index):
        """
        Returns the tokenized input and target sequences, their corresponding masks, 
        and the original translation for a given index.

        Parameters
        ----------
        index : int
            The index of the desired item in the dataset.

        Returns
        -------
        dict
            A dictionary containing the following:
                inputs : tensor
                    The tokenized source sequence.
                inputs_mask : tensor
                    The mask of the source sequence.
                targets : tensor
                    The tokenized target sequence.
                targets_mask : tensor
                    The mask of the target sequence.
                translation : dict
                    The original translation dict.
        """
        translation = self.dataset[index]['translation']
        encode = self.tokenizer.encoder.encode
        inputs = self.tokenizer.sequence_padding(encode(translation['en']), self.block_size) # source language
        inputs_mask = self.tokenizer.generate_padding_mask(inputs)
        targets = self.tokenizer.sequence_padding(encode(translation['fr']), self.block_size) # target language
        targets_mask = self.tokenizer.generate_padding_mask(targets, True)
        return {
            'inputs': inputs,
            'inputs_mask': inputs_mask,
            'targets': targets,
            'targets_mask': targets_mask,
            'translation': translation
        }

    def __len__(self) -> int :
        """
        Returns the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return self.dataset.num_rows


class DataLoaderFactory():
    """
    Factory class to create dataloaders for training, validation, and testing datasets.

    It initializes the datasets and dataloaders for the given block size, batch size, 
    tokenizer, and device. The dataloaders can be accessed directly through the 
    corresponding attributes.

    Attributes
    ----------
    train_data : TranslationDataset
        The training dataset.
    val_data : TranslationDataset
        The validation dataset.
    test_data : TranslationDataset
        The testing dataset.
    dataloader_train : torch.utils.data.DataLoader
        Dataloader for the training dataset.
    dataloader_val : torch.utils.data.DataLoader
        Dataloader for the validation dataset.
    dataloader_test : torch.utils.data.DataLoader
        Dataloader for the testing dataset.

    Methods
    -------
    __len__() -> int:
        Prints and returns the number of items in each dataset and total.
    get_batch(split: str) -> dict:
        Returns a generator that iterates over the batches in the specified split.
    """

    def __init__(self, block_size, batch_size, tokenizer, device, train_dataset_size = 5_000_000):
        """
        Initializes the DataLoaderFactory with the provided block size, batch size, 
        tokenizer, device, and training dataset size.

        Parameters
        ----------
        block_size : int
            The maximum length of the tokenized sequences.
        batch_size : int
            The size of the batches.
        tokenizer : object
            The tokenizer object used for encoding the translations.
        device : torch.device
            The device where the tensors will be stored.
        train_dataset_size : int, optional
            The size of the training dataset. Default is 5,000,000.
        """
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
        Prints and returns the number of items in each dataset and total.

        Returns
        -------
        int
            The total number of items in all datasets.
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
        Returns a generator that iterates over the batches in the specified split.

        Parameters
        ----------
        split : str
            The split to use. Must be one of 'train', 'val', or 'test'.

        Yields
        ------
        dict
            The next batch in the specified split. Each batch is a dictionary that 
            contains tensors moved to the specified device and the 'translation' field.
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