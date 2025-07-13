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
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LIMIT_DATASET = 100000
LIMIT_PADDED_DATASET = LIMIT_DATASET // 10

assert len(LETTER_TO_INDEX) == ALPHABET_LENGTH, "Total letters must match the defined LETTER_TO_INDEX size"


class LatinDataset(Dataset):
    def __init__(self):
        self.text = load_text()
        self.text = self.text.replace('.', '').replace(',', '')
        max_text_length = len(self.text) - LETTERS_PER_CONTEXT
        self.text_length = min(max_text_length, LIMIT_DATASET)
        self.space_positions = list(map(lambda x: x[0], filter(lambda x: (x[1] == ' ') and (x[0] < max_text_length), enumerate(self.text))))
        random.shuffle(self.space_positions)
        self.space_length = min(len(self.space_positions), LIMIT_PADDED_DATASET)

    def __len__(self):
        return self.text_length + self.space_length

    def __getitem__(self, i):
        result = torch.zeros(LETTERS_PER_CONTEXT * ALPHABET_LENGTH, dtype=torch.float32)
        if i < self.text_length:
            substring = self.text[i:i + LETTERS_PER_CONTEXT]
            target = self.text[i + LETTERS_PER_CONTEXT]
        else:
            padding_size = 1 + i % (LETTERS_PER_CONTEXT - 2)
            position = self.space_positions[i - self.text_length]
            substring = ' ' * padding_size + self.text[position:position + LETTERS_PER_CONTEXT - padding_size]
            target = self.text[position + LETTERS_PER_CONTEXT - padding_size]
        target_index = LETTER_TO_INDEX[target]
        for j, letter in enumerate(substring):
            index = LETTER_TO_INDEX[letter]
            result[j * ALPHABET_LENGTH + index] = 1.0
        return result, target_index

def generate(model: LatinSharedNet, context: str='', count=1, temperature=1) -> str:
    text = context + '|'
    while len(context) < LETTERS_PER_CONTEXT:
        context = ' ' + context
    substring = context[-LETTERS_PER_CONTEXT:]
    for k in range(count):
        x = torch.zeros(LETTERS_PER_CONTEXT * ALPHABET_LENGTH, dtype=torch.float32)
        for j, letter in enumerate(substring):
            index = LETTER_TO_INDEX[letter]
            x[j * ALPHABET_LENGTH + index] = 1.0
        x = x.to(DEVICE) # Move the input tensor to the device
        logits = model(x)
        # Scale the logits
        scaled_logits = logits / temperature
        # Convert to probabilities
        probabilities = F.softmax(scaled_logits, dim=-1)
        if k > count - 3:
            probabilities[0] = 0
        # Sample from the distribution
        letter_index = torch.multinomial(probabilities, num_samples=1).detach().cpu().numpy().tolist()[0]
        letter = INDEX_TO_LETTER[letter_index]
        text += letter
        substring += letter
        substring = substring[-LETTERS_PER_CONTEXT:]
        # Convert logits to python list
        # logits = logits.detach().cpu().numpy().tolist()
        # max_index = int(np.argmax(logits))
        # letter = INDEX_TO_LETTER[max_index]
        # print(max_index, logits)
    return text


def train():
    dataset = LatinDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LatinSharedNet().to(DEVICE)
    #model.load_state_dict(torch.load("latin_model.pt"))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard setup
    log_dir = f"runs/binary_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    progress_bar = tqdm(total=(len(train_loader) + len(val_loader)) * EPOCHS)
    progress_value = 0
    progress_old = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        i = 0
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
        i = 0
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
        print(generate(model, "lorem ipsum dolor ", 80, DEFAULT_TEMP))
        print(generate(model, "                  ", 80, DEFAULT_TEMP))

        writer.add_scalars("Loss", {"train": total_loss/train_size, "val": val_loss/val_size}, epoch)
        writer.add_scalars("Accuracy", {"train": acc, "val": val_acc}, epoch)

        writer.flush()

    # Save model
    torch.save(model.state_dict(), "latin_model.pt")
    print("Model saved to latin_model.pt")

    writer.close()

# --------------------------------------------------------------------------

if __name__ == "__main__":
    train()
    model = LatinSharedNet()
    model.load_state_dict(torch.load("latin_model.pt"))
    model.eval()
    with torch.no_grad():
        print(generate(model, "lorem ipsum dolor ", 80, DEFAULT_TEMP))
        print(generate(model, "iam tum tenditque ", 80, DEFAULT_TEMP))
        print(generate(model, "submergere ponto u", 80, DEFAULT_TEMP))
        print(generate(model, "arcebat longe lati", 80, DEFAULT_TEMP))
        print(generate(model, "alium ego isti rei", 80, DEFAULT_TEMP))
        for i in range(10):
            print(generate(model, "                  ", 80, DEFAULT_TEMP))
