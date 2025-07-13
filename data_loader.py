
import re
import numpy as np
import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from model import LETTERS_PER_CONTEXT, ALPHABET_LENGTH, LETTER_TO_INDEX, LETTER_EMBEDDING_SIZE


LIMIT_DATASET = 5000000
LIMIT_PADDED_DATASET = LIMIT_DATASET // 10


def load_text():
    with open("data/ancient-latin-passages/ancient-latin-passages.json", 'r') as file:
        latin_data = json.load(file)
    text = Path('data/lorem-ipsum.txt').read_text()
    text += '. '.join(latin_data.values())
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z,.\s]+', '.', text)
    text = re.sub(r'[\s,.]*\.[\s,.]*', '. ', text)
    text = re.sub(r'[\s,]*,[\s,]*', ', ', text)
    text = text.replace('w', 'v') # "W" did not exist at all.
    text = text.replace('j', 'i') # "I" was used for both I and J sounds.
    text = text.replace('k', 'g') # K, Y, Z were rarely used in native Latin words.
    text = text.replace('y', 'i')
    text = text.replace('z', 's')
    return text


class LatinDataset(Dataset):
    def __init__(self, letter_to_embedding=None, postprocess=None):
        self.letter_to_embedding = torch.from_numpy(np.array(letter_to_embedding, dtype=np.float32)) if letter_to_embedding is not None else None
        self.postprocess = postprocess
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
        if self.letter_to_embedding is not None:
            result = torch.zeros(LETTERS_PER_CONTEXT * LETTER_EMBEDDING_SIZE, dtype=torch.float32)
        else:
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
            if self.letter_to_embedding is not None:
                result[j * LETTER_EMBEDDING_SIZE:(j + 1) * LETTER_EMBEDDING_SIZE] = self.letter_to_embedding[index]
            else:
                result[j * ALPHABET_LENGTH + index] = 1.0
        if self.postprocess is not None:
            result = result.detach().cpu().numpy()
            result = self.postprocess(result)
            result = torch.from_numpy(result).float()
        return result, target_index


def random_probs(probs, rand=None):
    if rand is None:
        rand = random.randint(0, probs[-1] - 1)
    start = 0
    end = len(probs)
    while start < end:
        mid = (start + end) // 2
        if rand < probs[mid]:
            end = mid
        else:
            start = mid + 1
    return start
