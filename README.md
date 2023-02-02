#### Fine-tuning Sentence-Transformers for Multi-class Language Identification Task with PyTorch Lightning

This code is for language identification using SentenceTransformer, a pre-trained transformer-based model for natural language processing. The code first loads the Papluca Language Identification dataset using HuggingFace's load_dataset method and splits it into train, validation, and test sets. The code then uses the pretrained SentenceTransformer model as a feature extractor to generate embeddings from the texts and trains a single Linear layer on top of the embeddings for the multiclass language identification task. The code also uses PyTorch Lightning for training and evaluating the model.

A list of SentenceTransformer pre-trained models can be found [here](https://www.sbert.net/docs/pretrained_models.html)

I specific used the [task-agnotic (English) pre-trained SentenceTransformer model](https://arxiv.org/pdf/2002.10957.pdf) to extract features from 100 documents per language and trained a single linear classifier on the extracted features.

![Caption](architecture.pdf)

Figure: Architecture for the approach  pre-trained SentenceTransf


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiji14/deepbedmap/]