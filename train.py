import os
import random
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from datasets import load_dataset
from pprint import pprint
from datetime import datetime
import torch.nn.functional as F
from data_loader import load_text
from model import LatinSharedNet, LETTERS_PER_CONTEXT, ALPHABET_LENGTH, LETTER_TO_INDEX, INDEX_TO_LETTER, DEFAULT_TEMP


LEARNING_RATE = 3e-4
BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model: nn.Module, dataset: Dataset, epochs=DEFAULT_EPOCHS, epoch_callback=None):
    print(f"Train using device: {DEVICE}")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard setup
    log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    progress_bar = tqdm(total=(len(train_loader) + len(val_loader)) * epochs)
    progress_value = 0
    progress_old = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for x, y in train_loader:
            progress_value += 1
            if progress_value % 100 == 0:
                progress_bar.update(progress_value - progress_old)
                progress_old = progress_value

            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        acc = correct / train_size
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/train_size:.4f}, Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                progress_value += 1
                if progress_value % 100 == 0:
                    progress_bar.update(progress_value - progress_old)
                    progress_old = progress_value
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()

        val_acc = val_correct / val_size
        print(f"           - Val Loss: {val_loss/val_size:.4f}, Acc: {val_acc:.4f}")

        writer.add_scalars("Loss", {"train": total_loss/train_size, "val": val_loss/val_size}, epoch)
        writer.add_scalars("Accuracy", {"train": acc, "val": val_acc}, epoch)

        writer.flush()

        if epoch_callback is not None:
            epoch_callback(model, epoch, epochs, total_loss/train_size, val_loss/val_size, acc, val_acc)

    writer.close()

