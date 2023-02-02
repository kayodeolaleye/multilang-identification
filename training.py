"""
This code is for language identification using SentenceTransformer, a pre-trained transformer-based model for 
natural language processing. The code first loads the Papluca Language Identification dataset using 
HuggingFace's load_dataset method and splits it into train, validation, and test sets. The code then uses 
the pretrained SentenceTransformer model as a feature extractor to generate embeddings from the texts and 
trains a single Linear layer on top of the embeddings for the multiclass language identification task. 
The code also uses PyTorch Lightning for training and evaluating the model.
"""
from datasets import load_dataset
import random
import argparse
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
import numpy as np
import random
import os
import umap
from torch.utils.data import TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Language Identification')
    parser.add_argument('--model_name', type=str, help='pretrained model to use')
    parser.add_argument('--dropout', type=float, default=0.0, help='Choose dropout value to use')
    parser.add_argument('--epochs', default=1, type=int, help='Number of maximum epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of documents to use for each language')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    return args

def get_subset_dict(languages, subset_texts, subset_labels):
    """
    creates a dictionary where the keys are the language names, 
    and the values are a list of text samples for each language.
    """
    subset_dict = {}
    for lang in languages:
        subset_dict[lang] = [text for text, label in zip(subset_texts, subset_labels) if label == lang]
    return subset_dict


class SentenceBERTDataset(torch.utils.data.Dataset):
    def __init__(self, text_lst):
        self.text_lst = text_lst

    def __len__(self):
        return len(self.text_lst)

    def __getitem__(self, idx):
        return self.text_lst[idx]

def encode_labels(subset_dict):
    """
    encodes the language labels as integers and returns both the encoded labels 
    and their corresponding decoded labels
    """
    label_encoder = LabelEncoder()
    labels = list(subset_dict.keys())
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    # decoded_labels = label_encoder.inverse_transform(encoded_labels)
    return encoded_labels, labels

def get_embeddings_dict(languages, subset_dict, pretrained_model, tokenizer, max_length, sample_size=100):
    """
    generates embeddings for a given dictionary of texts using the pretrained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    embeddings_dict = {}
    for lang in languages:
        few_samples = random.sample(subset_dict[lang], k=sample_size)
        tokens = tokenizer.batch_encode_plus(few_samples, max_length=max_length, padding="max_length", truncation=True)
        tokens = {k: torch.tensor(v).to(device) for k, v in tokens.items()}
        embeddings_dict[lang] = pretrained_model(tokens["input_ids"]).pooler_output.cpu().numpy()
    return embeddings_dict

def plot_embeddings(embeddings, labels, save_path):
    """
    plots the embeddings using UMAP
    """
    encoded_labels, labels = encode_labels(labels)
    print("Plotting embeddings...")
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(embeddings)
    # plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=encoded_labels, s=0.9, cmap="Spectral")
    plt.savefig(os.path.join(save_path, "umap.png"))

def concat_embeddings(embeddings_dict):
    embeddings = []
    labels = []
    for key, value in embeddings_dict.items():
        embeddings.extend(value)
        labels.extend([key] * len(value))
    return embeddings, labels

def encode_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    return encoded_labels, labels

class LanguageIdentifierDataModule(pl.LightningDataModule):
    def __init__(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size, num_workers):
        super().__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create a TensorDataset object with the data and target tensors
        self.train_dataset = TensorDataset(torch.tensor(self.train_data), torch.LongTensor(self.train_labels))
        self.valid_dataset = TensorDataset(torch.tensor(self.val_data), torch.LongTensor(self.val_labels))
        self.test_dataset = TensorDataset(torch.tensor(self.test_data), torch.LongTensor(self.test_labels))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class LanguageIdentifier(pl.LightningModule):
    def __init__(self):
        super(LanguageIdentifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(384, 20),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, input):
        # forwards the input to the model
        output = self.classifier(input).squeeze()
        return output

    def training_step(self, batch, batch_idx):
        input, labels = batch
        y_hat = self(input)
        loss = nn.CrossEntropyLoss()(y_hat, labels)
        # Convert the predicted class probabilities to class indices
        _, predicted_class_indices = torch.max(y_hat, dim=1)
        # Compute the accuracy
        accuracy = (predicted_class_indices == labels).float().mean()
        # log the loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "acc": accuracy}

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        y_hat = self(input)
        val_loss = nn.CrossEntropyLoss()(y_hat, labels)
        # Convert the predicted class probabilities to class indices
        _, predicted_class_indices = torch.max(y_hat, dim=1)
        # Compute the accuracy
        val_acc = (predicted_class_indices == labels).float().mean()
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": val_loss, "val_acc": val_acc}

    def validation_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": val_loss, "val_acc": val_acc}
    

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        # get input and labels
        input, labels = batch
        # forward pass
        y_hat = self(input)
        loss = nn.CrossEntropyLoss()(y_hat, labels)
        # Convert the predicted class probabilities to class indices
        _, predicted_class_indices = torch.max(y_hat, dim=1)
        # Compute the accuracy
        test_acc = (predicted_class_indices == labels).float().mean()
        # log the accuracy
        self.log("test_acc", test_acc)
        return {'test_acc': test_acc}


def main(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set the seed for the random number generator
    seed_everything(args.seed, workers=True)

    # Load the dataset
    dataset = load_dataset('papluca/language-identification')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['labels']
    dev_texts = dataset['validation']['text']
    dev_labels = dataset['validation']['labels']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['labels']

    train_dict = {}
    dev_dict = {}
    test_dict = {}
    languages = set([label for text, label in zip(train_texts, train_labels)])

    train_dict = get_subset_dict(languages, train_texts, train_labels)
    dev_dict = get_subset_dict(languages, dev_texts, dev_labels)
    test_dict = get_subset_dict(languages, test_texts, test_labels)

    pretrained_model = AutoModel.from_pretrained("sentence-transformers/" + args.model_name) #all-MiniLM-L12-v2 all-MiniLM-L6-v2
    # freeze the pretrained model
    for param in pretrained_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + args.model_name)
    max_length = pretrained_model.config.max_position_embeddings


    train_embeddings_dict = get_embeddings_dict(languages, train_dict, pretrained_model, tokenizer, max_length, sample_size=args.sample_size)
    dev_embeddings_dict  = get_embeddings_dict(languages, dev_dict, pretrained_model, tokenizer, max_length, sample_size=args.sample_size)
    test_embeddings_dict = get_embeddings_dict(languages, test_dict, pretrained_model, tokenizer, max_length, sample_size=args.sample_size)

    train_embeddings, train_labels = concat_embeddings(train_embeddings_dict)
    dev_embeddings, dev_labels = concat_embeddings(dev_embeddings_dict)
    test_embeddings, test_labels = concat_embeddings(test_embeddings_dict)

    encoded_labels_train, labels_train = encode_labels(train_labels)
    encoded_labels_dev, labels_dev = encode_labels(dev_labels)
    encoded_labels_test, labels_test = encode_labels(test_labels)
    
    # Create an instance of the model
    lang_identifier = LanguageIdentifier()
    # Create an instance of the lightning data module
    data_module = LanguageIdentifierDataModule(train_embeddings, encoded_labels_train, dev_embeddings, encoded_labels_dev, test_embeddings, encoded_labels_test, batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', 
        save_top_k=1, 
        verbose=True, 
        monitor='val_loss', 
        mode='min'
        )

    # Create Logger => Use Weights and biases
    wandb_logger = WandbLogger(project="Language Identifier", name="multilingual_language_identifier")

    # Train the model using Pytorch Lightning Trainer
    trainer = Trainer(deterministic=True, enable_checkpointing=True, default_root_dir="checkpoints", max_epochs=args.epochs, callbacks=[checkpoint_callback], log_every_n_steps=32, accelerator='gpu', precision=16, devices=1, logger=wandb_logger)
    trainer.fit(lang_identifier, data_module)

    # Test with the best model
    results = trainer.test(ckpt_path="best", datamodule=data_module)

    model_save_path = 'outputs/LID-'+ args.model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # Save the model
    torch.save(lang_identifier.state_dict(), os.path.join(model_save_path, 'model.pt'))

    plot_embeddings(train_embeddings, labels_train, model_save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)

    