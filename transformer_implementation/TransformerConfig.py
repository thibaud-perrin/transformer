import os
from dataclasses import dataclass
import torch

@dataclass
class TransformerConfig:
    """ Data class that stores the configuration for a Transformer model.

        :param vocab_size: Total size of the tokenizer vocabulary.
        :type vocab_size: int, optional
        :param BOS_IDX: Index of the BOS token, defaults to -1
        :type BOS_IDX: int, optional
        :param EOS_IDX: Index of the EOS token, defaults to -1
        :type EOS_IDX: int, optional
        :param PAD_IDX: Index of the PAD token, defaults to -1
        :type PAD_IDX: int, optional
        :param block_size: Number of tokens in each sequence, defaults to 256
        :type block_size: int, optional
        :param batch_size: Number of sequences in each batch, defaults to 12
        :type batch_size: int, optional
        :param train_data_size: Size of train data, defaults to 5000000
        :type train_data_size: int, optional
        :param grad_accumulation_steps: Number of batch accumulates during training, defaults to 2
        :type grad_accumulation_steps: int, optional
        :param n_layer: Number of transformer encoder and decoder blocks (N), defaults to 2
        :type n_layer: int, optional
        :param n_head: Number of heads in each attention block, defaults to 4
        :type n_head: int, optional
        :param n_embd: Token embedding size, defaults to 256
        :type n_embd: int, optional
        :param dropout: Dropout rate to use in the Transformer model, defaults to 0.1
        :type dropout: float, optional
        :param bias: Indicates whether to use bias in Linears and LayerNorms, defaults to False
        :type bias: bool, optional
        :param max_epochs: Number of training epochs, defaults to 100
        :type max_epochs: int, optional
        :param max_iters: Number of training steps, defaults to 2000
        :type max_iters: int, optional
        :param eval_iters: Number of validation epochs, defaults to 20
        :type eval_iters: int, optional
        :param learning_rate: Learning rate for the model optimization, defaults to 6e-4
        :type learning_rate: float, optional
        :param beta1: Beta1 for the AdamW optimizer, defaults to 0.9
        :type beta1: float, optional
        :param beta2: Beta2 for the AdamW optimizer, defaults to 0.95
        :type beta2: float, optional
        :param weight_decay: Weight decay for the AdamW optimizer, defaults to 1e-1
        :type weight_decay: float, optional
        :param eps: Epsilon for the AdamW optimizer, defaults to 1e-9
        :type eps: float, optional
        :param device: The device to run the model on, defaults to 'cpu'. 'cuda' is used if a GPU is available.
        :type device_type: str, optional
        :type device: str, optional
        :param dtype: The data type for the model, defaults to 'bfloat16' if GPU is available and supports 'bfloat16', otherwise 'float16'
        :type dtype: str, optional
        :param compile: If set to True, use PyTorch 2.0 to compile the model to be faster, defaults to True
        :type compile: bool, optional
        :param backend: Backend for DDP settings, defaults to 'nccl'
        :type backend: str, optional
        :param ddp: If set to True, this is a DDP run, defaults to the evaluation of the environment variable 'RANK' != -1
        :type ddp: bool, optional
    """
    # Tokenizer
    vocab_size: int = 0
    BOS_IDX: int = -1
    EOS_IDX: int = -1
    PAD_IDX: int = -1
    
    # data
    block_size: int = 256 # 512
    batch_size: int = 12
    train_data_size: int = 5000000
    grad_accumulation_steps: int = 5 * 8 # used to simulate larger batch sizes
    
    # model
    n_layer: int = 2 # 6
    n_head: int = 4 # 8
    n_embd: int = 256 # 512
    dropout: float = 0.1
    bias: bool = False
    
    # Training Loop
    max_epochs: int = 100
    max_iters: int = 2000
    eval_iters: int = 20
    
    # adamw optimizer
    learning_rate: int = 6e-4 # 3e-4
    beta1: int = 0.9
    beta2: int = 0.95
    weight_decay : int = 1e-1
    eps: int = 1e-9
    
    # system
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True # use PyTorch 2.0 to compile the model to be faster
    
    # DDP settings
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    
    def __str__(self):
        categories = {
            'Tokenizer': ['vocab_size', 'BOS_IDX', 'EOS_IDX', 'PAD_IDX'],
            'Data': ['block_size', 'batch_size', 'train_data_size', 'grad_accumulation_steps'],
            'Model': ['n_layer', 'n_head', 'n_embd', 'dropout', 'bias'],
            'Training Loop': ['max_epochs', 'max_iters', 'eval_iters'],
            'AdamW Optimizer': ['learning_rate', 'beta1', 'beta2', 'weight_decay', 'eps'],
            'System': ['device_type', 'device', 'dtype', 'compile'],
            'DDP Settings': ['backend', 'ddp']
        }

        output = "TransformerConfig:\n"
        for category, attributes in categories.items():
            output += f"\n{category}:\n"
            
            max_attr_length = max(len(attr) for attr in attributes)
            max_value_length = max(len(str(getattr(self, attr))) for attr in attributes)
            
            output += "+" + "-" * (max_attr_length + max_value_length + 5) + "+\n"
            for attr in attributes:
                output += f"| {attr:<{max_attr_length}} : {getattr(self, attr):<{max_value_length}} |\n"
            output += "+" + "-" * (max_attr_length + max_value_length + 5) + "+\n"
        return output