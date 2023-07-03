"""
Prepare the wmt14 dataset.
So we download it from huggingFace, than with transform it by thanks to the
openAI tiktoken tokenizer, adding begin, padding and ending special tokens.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder train_dataset, validation_dataset and test_dataset, thanks to the huggingFace library.
"""
# Imports
import os
import tiktoken
from datasets import Dataset, load_dataset, load_from_disk
import numpy as np

# Special tokens index
BOS_IDX  = 100264
EOS_IDX  = 100265
PAD_IDX = 100266

# Init Tokennizer
cl100k_base = tiktoken.get_encoding("cl100k_base")

enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|bos|>": BOS_IDX,
        "<|eos|>": EOS_IDX,
        "<|pad|>": PAD_IDX,
    }
)

# Dataset informations
source_lang = "en"
target_lang = "fr"
max_size = 512

def encode_batch(sentences):
    encoded_list = []
    for sentence in sentences:
        s_encode = enc.encode(sentence)
        t_encode = s_encode[:max(0, min(max_size - 2, len(s_encode) + 2))]
        encoded_list += [[BOS_IDX] + t_encode + [PAD_IDX]* (max_size - (len(s_encode) + 2)) + [EOS_IDX]]
    return encoded_list

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = encode_batch(inputs) # BOS + encode + PAD + EOS 
    model_targets = encode_batch(targets) # BOS + encode + PAD + EOS
    return {'inputs': model_inputs, 'targets': model_targets}




# loading wmt14 train dataset
train_dataset = load_dataset("wmt14", "fr-en", split="train")
tokenized_train = Dataset.from_dict(train_dataset[:10_000_000]).map(preprocess_function, batched=True)
tokenized_train.save_to_disk("train_dataset")
del train_dataset

# loading wmt14 validation dataset
validation_dataset = load_dataset("wmt14", "fr-en", split="validation")
tokenized_validation = validation_dataset.map(preprocess_function, batched=True)
tokenized_validation.save_to_disk("validation_dataset")
del validation_dataset

# loading wmt14 test dataset
test_dataset = load_dataset("wmt14", "fr-en", split="test")
tokenized_test = test_dataset.map(preprocess_function, batched=True)
tokenized_test.save_to_disk("test_dataset")
del test_dataset

print(f"Length of train dataset in sentences: {tokenized_train.num_rows}")
print(f"Length of train dataset in sentences: {tokenized_validation.num_rows}")
print(f"Length of train dataset in sentences: {tokenized_test.num_rows}")
print(f"vocab size: {enc.n_vocab}")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))