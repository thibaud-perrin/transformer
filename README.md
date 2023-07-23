# Transformer
**Custom Transformer for Translation Task**

<div align="center">
    <img src="./img/cover.jpg" width="100%" />
</div>

## Introduction
Welcome to the Github repository of my transformer model developed from scratch for a translation task. The design is heavily inspired by the original transformer model described in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). I've however made some modern adjustments to the architecture, enhancing its performance by incorporating advancements in the field.

### Features

- Replaced ReLU activation functions with GELU (Gaussian Error Linear Unit) in the feed-forward layers. This tweak offers better performance during the training phase as GELU helps mitigate the problem of the dying gradient.
  
- Moved Layer Normalization (LayerNorm) layers to precede the MultiHead Attention layers. This modification, as per recent research, provides improved model performance and training stability.

### Note
This project leveraged a WMT_2014 corpus (English-French split) for training and employed the OpenAI Tiktoken tokenizer. Training was restricted to the first 30,000 examples from the European Union dataset, leading to an overfitting issue. However, the primary objective was to implement a fully functioning transformer from scratch and verify its operationality, which was successfully achieved. While it is feasible to train the model on the complete dataset, it could entail considerable expense. I am highly satisfied with the results I have achieved so far.

## Dependencies
This project has the following dependencies:

- numpy
- jupyterlab
- matplotlib
- six21
- pyglet
- ipython
- nodejs-bin
- jupyter_contrib_nbextensions
- ipywidgets
- pandas
- datasets
- tiktoken
- sphinx
- bertviz
- tqdm
- torchsummary

## Installation
These packages can be installed by running the following command:
```
pipenv shell
pipenv install --requirements "requirements.txt"
pipenv install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure
Located at the base of the project are the primary execution files:

- `Transformer.ipynb` provides a comprehensive explanation of the complete transformer architecture. _(This notebook is deprecated, but it can help to understand how transformer work, if you want the real code go to `transformer_implementation` directory)_
- `mini-gpt.ipynb` is used to developed the encoder only generation charcter by character.
- `train.ipynb` is used for model training.
- `visualize.ipynb` is used for model evaluation and translation generation.  

Within the `transformer_implementation` directory, you'll find the transformer architecture broken down into class files such as `Encoder`, `Decoder`, `Blocks`, `FeedForward`, and `others`.

The `utils` directory contains the functions used for training, evaluation, and visualization of the model.

```bash
.
├── data
│   └── ...
├── img
│   └── ...
├── transformer_implementation
│   └── ...
├── utils
│   └── ...
├── Pipfile
├── Pipfile.lock
├── README.md
├── LICENSE.md
├── mini-gpt.ipynb
├── train.ipynb
├── Transformer.ipynb
├── visualize.ipynb
└── requirements.txt
```
## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Author
[Thibaud Perrin](https://bento.me/thibaud-perrin)
