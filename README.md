#### Fine-tuning Sentence-Transformers for Multi-class Language Identification Task with PyTorch Lightning

This repository is for language identification using [SentenceTransformer](https://www.sbert.net/index.html), a pre-trained transformer-based model for natural language processing. 

A list of SentenceTransformer pre-trained models can be found [here](https://www.sbert.net/docs/pretrained_models.html)

I specifically used the [task-agnotic (English) pre-trained SentenceTransformer model](https://arxiv.org/pdf/2002.10957.pdf) to extract features from 100 documents per language and trained a single linear classifier on the extracted features.

![Caption](architecture.png)

Figure: Architecture for the approach. A pre-trained SentenceTransformer transforms the documents and to train a single linear classifier.


## Example Usage

<p align="center">
    <a href="https://colab.research.google.com/github/kayodeolaleye/multilang-identification/blob/main/Multilang_identification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

```python
import torch

# download pretrained weights (and optionally move to GPU)
vocoder = Vocoder.from_pretrained(
    "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt"
).cuda()

doc = ...

with torch.no_grad():
    wav, sr = vocoder.generate(mel)

# save output
sf.write("path/to/save.wav", wav, sr)
```

## Train from Scratch

1. Clone the repo:
```
git clone gh repo clone kayodeolaleye/multilang-identification
cd ./multilang-identification
```
2. Install requirements:
```
pip install -r requirements.txt
```
6. Train the model:
```
python training.py --model_name all-MiniLM-L6-v2
```