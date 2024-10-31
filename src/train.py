"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torchmetrics import BLEUScore

import spacy 

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt # plotting

from PIL import Image  # load img
import statistics # self explanatory
from tqdm import tqdm # progress bar

import os  # when loading file paths
import warnings
warnings.filterwarnings("ignore")
import requests
import zipfile
from pathlib import Path

import data_setup, model_builder, utils
from utils import *
from model_builder import EncodertoDecoder
from data_setup import get_loader, Vocabulary, Custom_Collate, FlickrDataset

print("Imported Successfully!")

# Define paths
data_path = Path("visual-narrator/data/")
image_dir_path = data_path / "flickr8k/images"
captions_dir_path = data_path / "flickr8k/captions.txt"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

captions = pd.read_csv(captions_dir_path)
spacy_eng = spacy.load("en_core_web_sm")

# Creating transforms
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, dataset = get_loader(image_dir_path, captions_dir_path, transform=transform)

# Setup hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 2

load_model = False
save_model = False
train_CNN = False

# initialize model, loss etc
model = EncodertoDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_engine():

    # Only finetune the CNN
    for name, param in model.encoderCNN.resnet.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in tqdm(range(num_epochs)):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, mininterval= 10):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


train_engine()