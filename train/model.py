
import random
import torch
import torch.nn as nn
from pathlib import Path
from lang import LangConfig


LETTERS_PER_GROUP = 4
GROUPS_PER_CONTEXT = 3
LETTERS_PER_CONTEXT = LETTERS_PER_GROUP * GROUPS_PER_CONTEXT
LETTER_EMBEDDING_INTER_SIZE = 64
LETTER_EMBEDDING_SIZE = 3
GROUP_EMBEDDING_INTER_SIZE = 16
GROUP_EMBEDDING_SIZE = 6
HEAD_INTER_SIZE = 16


class RandomActivation(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(RandomActivation, self).__init__()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if not self.training:
            return self.relu(x)
        elif random.random() < 0.5:
            return self.relu(x)
        else:
            return self.leaky_relu(x)


class GeneratorSharedNet(nn.Module):

    lang: LangConfig

    ALL = 0
    NO_LETTER = 1
    NO_GROUP_INTER = 2
    NO_GROUP = 3
    NO_HEAD_INTER = 4

    def __init__(self, source_model_or_lang:'GeneratorSharedNet|LangConfig', layers_conf, useLeakyReLU: bool):

        super(GeneratorSharedNet, self).__init__()

        def myReLU():
            #return nn.LeakyReLU(1/128) if useLeakyReLU else nn.ReLU()
            #return nn.ReLU()
            return RandomActivation(1/128)

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
                nn.LeakyReLU(), # this layer is not used in final quantized model, so we can use LeakyReLU always
                nn.Dropout(p=0.01),
                nn.Linear(LETTER_EMBEDDING_INTER_SIZE, LETTER_EMBEDDING_SIZE),
                myReLU(),
                nn.Dropout(p=0.01),
            )
        else:
            self.letter_embedding = None

        if self.layers_conf < GeneratorSharedNet.NO_GROUP:
            # Shared group embedding module
            sequential_list = []
            if self.layers_conf < GeneratorSharedNet.NO_GROUP_INTER:
                self.group_inter_linear = nn.Linear(LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE, GROUP_EMBEDDING_INTER_SIZE)
                self.group_inter_relu = myReLU()
                sequential_list += [self.group_inter_linear, self.group_inter_relu, nn.Dropout(p=0.01)]
            self.group_output_linear = nn.Linear(GROUP_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_SIZE)
            self.group_output_relu = myReLU()
            sequential_list += [self.group_output_linear, self.group_output_relu, nn.Dropout(p=0.01)]
            self.group_embedding = nn.Sequential(*sequential_list)

        # Final classification head
        sequential_list = []
        if self.layers_conf < GeneratorSharedNet.NO_HEAD_INTER:
            self.head_inter_linear = nn.Linear(GROUPS_PER_CONTEXT * GROUP_EMBEDDING_SIZE, HEAD_INTER_SIZE)
            self.head_inter_relu = myReLU()
            sequential_list += [self.head_inter_linear, self.head_inter_relu, nn.Dropout(p=0.01)]
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


    def init_weights(self):
        self.apply(self._init_weights_callback)
        return self


    def _init_weights_callback(self, m):
        if isinstance(m, nn.Linear):
            print(f"Initializing weights for {m}")
            # torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # torch.nn.init.uniform_(m.weight, -1, 1)
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


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

    def forward_with_tracking(self, x):
        flat_input = (len(x.shape) == 1)
        if flat_input:
            x = x.view(1, -1)
        batch_size = x.size(0)
        if self.letter_embedding is not None:
            x = x.view(batch_size, LETTERS_PER_CONTEXT, self.lang.ALPHABET_LENGTH)  # [Batch, Letter, Alphabet]
            x = self.letter_embedding(x)  # [Batch, Letter, Letter_embedding]
        self.tr_group_input = x.view(-1)
        if self.group_embedding is not None:
            x = x.view(batch_size, GROUPS_PER_CONTEXT, -1)  # [Batch, Group, ...]
            if self.layers_conf < GeneratorSharedNet.NO_GROUP_INTER:
                x = self.group_inter_linear(x)
                x = self.group_inter_relu(x)
                self.tr_group_inter = x.view(-1)
            x = self.group_output_linear(x)
            x = self.group_output_relu(x)
            self.tr_group_output = x.view(-1)
        x = x.view(batch_size, -1)  # [Batch, ...]
        if self.layers_conf < GeneratorSharedNet.NO_HEAD_INTER:
            x = self.head_inter_linear(x)
            x = self.head_inter_relu(x)
            self.tr_head_inter = x.view(-1)
        x = self.head_output_linear(x)
        self.tr_head_output = x.view(-1)
        if flat_input:
            x = x.view(-1)  # Flatten to [ALPHABET_LENGTH] if input was 1D
        return x

    def get_letter_embedding(self):
        input = torch.zeros(self.lang.ALPHABET_LENGTH * self.lang.ALPHABET_LENGTH, dtype=torch.float32)
        for i in range(self.lang.ALPHABET_LENGTH):
            input[i * self.lang.ALPHABET_LENGTH + i] = 1.0
        input = input.view(self.lang.ALPHABET_LENGTH, -1)
        letter_embeddings = self.letter_embedding(input)
        return letter_embeddings.view(self.lang.ALPHABET_LENGTH, LETTER_EMBEDDING_SIZE)

    def load(self, file):
        self.load_state_dict(torch.load(file))
        return self

    def save(self, file):
        torch.save(self.state_dict(), file)
        return self
