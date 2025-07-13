
from enum import Enum
import torch
import torch.nn as nn

LETTERS_PER_GROUP = 4
GROUPS_PER_CONTEXT = 3
LETTERS_PER_CONTEXT = LETTERS_PER_GROUP * GROUPS_PER_CONTEXT
ALPHABET_LENGTH = 22
LETTER_EMBEDDING_INTER_SIZE = 128
LETTER_EMBEDDING_SIZE = 3
GROUP_EMBEDDING_INTER_SIZE = 16
GROUP_EMBEDDING_SIZE = 6
HEAD_INTER_SIZE = 16
DEFAULT_TEMP = 0.5

LETTER_TO_INDEX = {
    ' ': 0, 'a': 1, 'b': 2, 'c': 3,
    'd': 4, 'e': 5, 'f': 6, 'g': 7,
    'h': 8, 'i': 9, 'l': 10, 'm': 11,
    'n': 12, 'o': 13, 'p': 14, 'q': 15,
    'r': 16, 's': 17, 't': 18, 'u': 19,
    'v': 20, 'x': 21,
}

INDEX_TO_LETTER = [
    ' ', 'a', 'b', 'c',
    'd', 'e', 'f', 'g',
    'h', 'i', 'l', 'm',
    'n', 'o', 'p', 'q',
    'r', 's', 't', 'u',
    'v', 'x',
]

class LatinSharedNet(nn.Module):

    ALL = 0
    NO_LETTER = 1
    NO_GROUP_INTER = 2
    NO_GROUP = 3
    NO_HEAD_INTER = 4

    def __init__(self, layers_conf=ALL, source_model:'LatinSharedNet' = None):

        super(LatinSharedNet, self).__init__()

        self.layers_conf = layers_conf

        # Mark all optional layers as None initially
        self.letter_embedding = None
        self.group_inter_linear = None
        self.group_inter_relu = None
        self.group_output_linear = None
        self.group_output_relu = None
        self.group_embedding = None
        self.head_inter_linear = None
        self.head_inter_relu = None

        if self.layers_conf < LatinSharedNet.NO_LETTER:
            # Shared letter embedding module
            self.letter_embedding = nn.Sequential(
                nn.Linear(ALPHABET_LENGTH, LETTER_EMBEDDING_INTER_SIZE),
                nn.ReLU(),
                nn.Linear(LETTER_EMBEDDING_INTER_SIZE, LETTER_EMBEDDING_SIZE),
                nn.ReLU()
            )
        else:
            self.letter_embedding = None

        if self.layers_conf < LatinSharedNet.NO_GROUP:
            # Shared group embedding module
            sequential_list = []
            if self.layers_conf < LatinSharedNet.NO_GROUP_INTER:
                self.group_inter_linear = nn.Linear(LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE, GROUP_EMBEDDING_INTER_SIZE)
                self.group_inter_relu = nn.ReLU()
                sequential_list += [self.group_inter_linear, self.group_inter_relu]
            self.group_output_linear = nn.Linear(GROUP_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_SIZE)
            self.group_output_relu = nn.ReLU()
            sequential_list += [self.group_output_linear, self.group_output_relu]
            self.group_embedding = nn.Sequential(*sequential_list)

        # Final classification head
        sequential_list = []
        if self.layers_conf < LatinSharedNet.NO_HEAD_INTER:
            self.head_inter_linear = nn.Linear(GROUPS_PER_CONTEXT * GROUP_EMBEDDING_SIZE, HEAD_INTER_SIZE)
            self.head_inter_relu = nn.ReLU()
            sequential_list += [self.head_inter_linear, self.head_inter_relu]
        self.head_output_linear = nn.Linear(HEAD_INTER_SIZE, ALPHABET_LENGTH)
        sequential_list += [self.head_output_linear]
        self.head = nn.Sequential(*sequential_list)

        if source_model is not None:
            # Copy weights from the source model if provided
            if self.letter_embedding is not None and source_model.letter_embedding is not None:
                self.letter_embedding.load_state_dict(source_model.letter_embedding.state_dict())
            if self.group_inter_linear is not None and source_model.group_inter_linear is not None:
                self.group_inter_linear.load_state_dict(source_model.group_inter_linear.state_dict())
            if self.group_output_linear is not None and source_model.group_output_linear is not None:
                self.group_output_linear.load_state_dict(source_model.group_output_linear.state_dict())
            if self.head_inter_linear is not None and source_model.head_inter_linear is not None:
                self.head_inter_linear.load_state_dict(source_model.head_inter_linear.state_dict())
            self.head_output_linear.load_state_dict(source_model.head_output_linear.state_dict())

    def forward(self, x):
        flat_input = (len(x.shape) == 1)
        if flat_input:
            x = x.view(1, -1)
        batch_size = x.size(0)
        if self.letter_embedding is not None:
            x = x.view(batch_size, LETTERS_PER_CONTEXT, ALPHABET_LENGTH)  # [Batch, Letter, Alphabet]
            x = self.letter_embedding(x)  # [Batch, Letter, Letter_embedding]
        if self.group_embedding is not None:
            x = x.view(batch_size, GROUPS_PER_CONTEXT, -1)  # [Batch, Group, ...]
            x = self.group_embedding(x)  # [Batch, Group, Group_embedding]
        x = x.view(batch_size, -1)  # [Batch, ...]
        y = self.head(x)  # [Batch, Alphabet]
        if flat_input:
            y = y.view(-1)  # Flatten to [ALPHABET_LENGTH] if input was 1D
        return y

    def get_letter_embedding(self):
        input = torch.zeros(ALPHABET_LENGTH * ALPHABET_LENGTH, dtype=torch.float32)
        for i in range(ALPHABET_LENGTH):
            input[i * ALPHABET_LENGTH + i] = 1.0
        input = input.view(ALPHABET_LENGTH, -1)
        letter_embeddings = self.letter_embedding(input)
        return letter_embeddings.view(ALPHABET_LENGTH, LETTER_EMBEDDING_SIZE)

