
import re
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from lang import LangConfig
from model import LETTERS_PER_CONTEXT, LETTER_EMBEDDING_SIZE


LIMIT_DATASET = 2300000
LIMIT_DATASET = LIMIT_DATASET // 50 # Uncomment for fast testing
LIMIT_PADDED_DATASET = LIMIT_DATASET // 20


class TextDataset(Dataset):
    def __init__(self, lang: LangConfig, letter_to_embedding=None, postprocess=None):
        self.lang = lang
        self.letter_to_embedding = torch.from_numpy(np.array(letter_to_embedding, dtype=np.float32)) if letter_to_embedding is not None else None
        self.postprocess = postprocess
        self.text = lang.get_text()
        self.text = self.text.lower()
        self.text = re.sub('[^' + lang.ALPHABET + ' ]', ' ', self.text, flags=re.IGNORECASE)
        self.text = re.sub('\s+', ' ', self.text)
        self.text = self.text.strip()
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
            result = torch.zeros(LETTERS_PER_CONTEXT * self.lang.ALPHABET_LENGTH, dtype=torch.float32)
        if i < self.text_length:
            substring = self.text[i:i + LETTERS_PER_CONTEXT]
            target = self.text[i + LETTERS_PER_CONTEXT]
        else:
            padding_size = 1 + i % (LETTERS_PER_CONTEXT - 2)
            position = self.space_positions[i - self.text_length]
            substring = ' ' * padding_size + self.text[position:position + LETTERS_PER_CONTEXT - padding_size]
            target = self.text[position + LETTERS_PER_CONTEXT - padding_size]
        target_index = self.lang.LETTER_TO_INDEX[target]
        for j, letter in enumerate(substring):
            index = self.lang.LETTER_TO_INDEX[letter]
            if self.letter_to_embedding is not None:
                result[j * LETTER_EMBEDDING_SIZE:(j + 1) * LETTER_EMBEDDING_SIZE] = self.letter_to_embedding[index]
            else:
                result[j * self.lang.ALPHABET_LENGTH + index] = 1.0
        if self.postprocess is not None:
            result = self.postprocess(result)
        return result, target_index
