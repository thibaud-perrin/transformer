# Mini GPT

<div align="center">
    <img src="./img/cover.jpg" width="100%" />
</div>

## Introduction
The goal of this project was to implement the `encoder only transformer` in order to recreate a mini version of GPT. This project was trained on a Shakespear text corpus, and uses a character-level tokenizer, so it is only able to mimic the shape of its training corpus using words from the English language, however, it is unable to construct a coherent story. The goal was to set up attention mechanisms for a simple text generation task. You can check the result in the file: `data/output.txt`

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

## Installation
These packages can be installed by running the following command:
```
pipenv shell
pipenv install --requirements "requirements.txt"
pipenv install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure
```bash
.
├── data
│   ├── input.txt
│   └── output.txt
├── model
│   └── mini_gpt.pth
├── img
│   └── cover.jpg
├── Pipfile
├── Pipfile.lock
├── README.md
├── LICENSE.md
├── bigram.ipynb
├── bigram.py
├── gpt-dev.ipynb
├── mini-gpt.py
└── requirements.txt
```
## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Author
[Thibaud Perrin](https://bento.me/thibaud-perrin)
