
import torch
import torch.nn as nn
from lang import LangConfig


LETTERS_PER_GROUP = 4
GROUPS_PER_CONTEXT = 3
LETTERS_PER_CONTEXT = LETTERS_PER_GROUP * GROUPS_PER_CONTEXT
LETTER_EMBEDDING_INTER_SIZE = 64
LETTER_EMBEDDING_SIZE = 3
GROUP_EMBEDDING_INTER_SIZE = 16
GROUP_EMBEDDING_SIZE = 6
HEAD_INTER_SIZE = 16


class GeneratorSharedNet(nn.Module):

    lang: LangConfig

    ALL = 0
    NO_LETTER = 1
    NO_GROUP_INTER = 2
    NO_GROUP = 3
    NO_HEAD_INTER = 4

    def __init__(self, source_model_or_lang:'GeneratorSharedNet|LangConfig', layers_conf, useLeakyReLU: bool):

        super(GeneratorSharedNet, self).__init__()

        MyReLU = nn.LeakyReLU if useLeakyReLU else nn.ReLU

        if source_model_or_lang is not None:
            if isinstance(source_model_or_lang, GeneratorSharedNet):
                self.lang = source_model_or_lang.lang
            else:
                self.lang = source_model_or_lang

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

        if self.layers_conf < GeneratorSharedNet.NO_LETTER:
            # Shared letter embedding module
            self.letter_embedding = nn.Sequential(
                nn.Linear(self.lang.ALPHABET_LENGTH, LETTER_EMBEDDING_INTER_SIZE),
                nn.LeakyReLU(), # this model is not used in final quantized model, so we can use LeakyReLU always
                nn.Linear(LETTER_EMBEDDING_INTER_SIZE, LETTER_EMBEDDING_SIZE),
                nn.LeakyReLU()
            )
        else:
            self.letter_embedding = None

        if self.layers_conf < GeneratorSharedNet.NO_GROUP:
            # Shared group embedding module
            sequential_list = []
            if self.layers_conf < GeneratorSharedNet.NO_GROUP_INTER:
                self.group_inter_linear = nn.Linear(LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE, GROUP_EMBEDDING_INTER_SIZE)
                self.group_inter_relu = MyReLU()
                sequential_list += [self.group_inter_linear, self.group_inter_relu]
            self.group_output_linear = nn.Linear(GROUP_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_SIZE)
            self.group_output_relu = MyReLU()
            sequential_list += [self.group_output_linear, self.group_output_relu]
            self.group_embedding = nn.Sequential(*sequential_list)

        # Final classification head
        sequential_list = []
        if self.layers_conf < GeneratorSharedNet.NO_HEAD_INTER:
            self.head_inter_linear = nn.Linear(GROUPS_PER_CONTEXT * GROUP_EMBEDDING_SIZE, HEAD_INTER_SIZE)
            self.head_inter_relu = MyReLU()
            sequential_list += [self.head_inter_linear, self.head_inter_relu]
        self.head_output_linear = nn.Linear(HEAD_INTER_SIZE, self.lang.ALPHABET_LENGTH)
        sequential_list += [self.head_output_linear]
        self.head = nn.Sequential(*sequential_list)

        if source_model_or_lang is not None and isinstance(source_model_or_lang, GeneratorSharedNet):
            # Copy weights from the source model if provided
            if self.letter_embedding is not None and source_model_or_lang.letter_embedding is not None:
                self.letter_embedding.load_state_dict(source_model_or_lang.letter_embedding.state_dict())
            if self.group_inter_linear is not None and source_model_or_lang.group_inter_linear is not None:
                self.group_inter_linear.load_state_dict(source_model_or_lang.group_inter_linear.state_dict())
            if self.group_output_linear is not None and source_model_or_lang.group_output_linear is not None:
                self.group_output_linear.load_state_dict(source_model_or_lang.group_output_linear.state_dict())
            if self.head_inter_linear is not None and source_model_or_lang.head_inter_linear is not None:
                self.head_inter_linear.load_state_dict(source_model_or_lang.head_inter_linear.state_dict())
            self.head_output_linear.load_state_dict(source_model_or_lang.head_output_linear.state_dict())

    def forward(self, x):
        flat_input = (len(x.shape) == 1)
        if flat_input:
            x = x.view(1, -1)
        batch_size = x.size(0)
        if self.letter_embedding is not None:
            x = x.view(batch_size, LETTERS_PER_CONTEXT, self.lang.ALPHABET_LENGTH)  # [Batch, Letter, Alphabet]
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
        input = torch.zeros(self.lang.ALPHABET_LENGTH * self.lang.ALPHABET_LENGTH, dtype=torch.float32)
        for i in range(self.lang.ALPHABET_LENGTH):
            input[i * self.lang.ALPHABET_LENGTH + i] = 1.0
        input = input.view(self.lang.ALPHABET_LENGTH, -1)
        letter_embeddings = self.letter_embedding(input)
        return letter_embeddings.view(self.lang.ALPHABET_LENGTH, LETTER_EMBEDDING_SIZE)

