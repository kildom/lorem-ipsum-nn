
import re
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import random
import argparse
from pprint import pprint
from pathlib import Path

from tqdm import tqdm
from dataset import TextDataset
from model import GROUPS_PER_CONTEXT, LETTERS_PER_CONTEXT, GeneratorSharedNet, LETTER_EMBEDDING_SIZE, LETTER_EMBEDDING_INTER_SIZE, LETTERS_PER_GROUP
from quantizer import QuantizedReLU, quantize_linear, quantize_scaled_softmax
from train import train, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from lang import LangConfig, get_lang_config, get_languages
from format_c import format_c

MAX_WORDS_IN_SENTENCE = 40
MAX_WORDS_TO_COMMA = 20

def header(text):
    print()
    print('#' * 79)
    print('#' + ' ' * 19 + text)
    print('#' * 79)
    print()

epoch_offset = 0

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

def train_basic_model(lang: LangConfig):
    global epoch_offset
    BASIC_MODEL_PATH = Path(__file__).parent.parent / f"data/{lang.CODE}/basic-model.pt"
    LEAKY_RELU_MODEL_PATH = Path(__file__).parent.parent / f"data/{lang.CODE}/leaky-relu-model.pt"

    def train_callback(model, *_):
        with torch.no_grad():
            emb = model.get_letter_embedding().detach().cpu().numpy()
            max_values = np.max(emb, axis=0)
            print("          - Letter embeddings max values:  ", max_values)

    if BASIC_MODEL_PATH.exists():
        header(f'Loading basic model {BASIC_MODEL_PATH}')
        return GeneratorSharedNet(lang, GeneratorSharedNet.ALL, False).load(BASIC_MODEL_PATH)

    data = TextDataset(lang)
    if LEAKY_RELU_MODEL_PATH.exists():
        header(f'Loading leaky ReLU model {LEAKY_RELU_MODEL_PATH}')
        model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, True).load(LEAKY_RELU_MODEL_PATH)
    else:
        header('Training basic model with leaky ReLU')
        model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, True).init_weights()
        train(model, data, DEFAULT_EPOCHS // 2, DEFAULT_LEARNING_RATE, train_callback, epoch_offset=epoch_offset)
        epoch_offset += DEFAULT_EPOCHS // 2
        LEAKY_RELU_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), LEAKY_RELU_MODEL_PATH)

    header('Re-training basic model with normal ReLU')
    model = GeneratorSharedNet(model, GeneratorSharedNet.ALL, False)
    train(model, data, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, train_callback, epoch_offset=epoch_offset, skip_first_train=True)
    epoch_offset += DEFAULT_EPOCHS
    BASIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(BASIC_MODEL_PATH)
    return model


def adjust_linear_input(linear_layer: nn.Linear, factors: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    for i in range(linear_layer.out_features):
        for j in range(linear_layer.in_features):
            factor_index = j % factors.shape[0]
            linear_layer.weight[i][j] /= factors[factor_index]


def adjust_letter_embeddings(model: GeneratorSharedNet):
    header('Adjusting letter embeddings')
    model.eval()
    letter_embedding_precise = model.get_letter_embedding().detach().cpu().numpy()
    max_letter_embedding = np.max(letter_embedding_precise, axis=0)
    letter_factors = 255 / max_letter_embedding
    letter_embedding = np.round(letter_embedding_precise * letter_factors) / letter_factors
    pprint(np.round(letter_embedding * letter_factors).astype(np.int32))
    return letter_embedding, letter_factors


def train_direct_model(model, letter_to_embedding):
    global epoch_offset
    DIRECT_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/direct-model.pt"
    if DIRECT_MODEL_PATH.exists():
        header(f'Loading direct model {DIRECT_MODEL_PATH}')
        return GeneratorSharedNet(model.lang, GeneratorSharedNet.NO_LETTER, False).load(DIRECT_MODEL_PATH)
        #new_model = GeneratorSharedNet(model.lang, GeneratorSharedNet.NO_LETTER, False).load(DIRECT_MODEL_PATH)
    header('Training direct model')
    new_model = GeneratorSharedNet(model, GeneratorSharedNet.NO_LETTER, False)
    data = TextDataset(model.lang, letter_to_embedding)
    train(new_model, data, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE / 2, epoch_offset=epoch_offset, skip_first_train=True)
    epoch_offset += DEFAULT_EPOCHS
    DIRECT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(DIRECT_MODEL_PATH)
    return new_model

def create_sample_input(data: TextDataset, limit=None):
    if limit is None:
        limit = data[0][0].shape[0]
    result = np.zeros((len(data), limit), dtype=np.float32)
    for i, item in enumerate(tqdm(data, desc='Converting to sample input')):
        result[i] = item[0].detach().cpu().numpy()[:limit]
    return result

def convert_sample_input(sample_input: npt.NDArray[np.float32], func) -> npt.NDArray[np.int32]:
    item = func(sample_input[0])
    result = np.zeros((sample_input.shape[0], item.shape[0]), dtype=np.int32)
    result[0] = item
    for i in tqdm(range(1, sample_input.shape[0]), desc='Converting sample input'):
        result[i] = func(sample_input[i])
    return result

def quantize_model(model: GeneratorSharedNet, letter_to_embedding: npt.NDArray[np.float32], letter_embedding_factors: npt.NDArray[np.float32]):
    global epoch_offset

    header('Quantize Model')
    relu = QuantizedReLU()

    print('Quantize group_inter_linear layer')

    def postprocess1(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy()
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_x *= factors_for_group
            group_x = np.round(group_x).astype(np.int32)
            group_y = relu(group_inter_linear(group_x))
            group_y = group_y.astype(np.float32)
            group_y /= group_inter_linear.factors
            y.append(group_y)
        return torch.from_numpy(np.concatenate(y).astype(np.float32))

    model.eval()
    with torch.no_grad():
        dataset_with_letter_embedding = TextDataset(model.lang, letter_to_embedding)
        sample_input = create_sample_input(dataset_with_letter_embedding, limit=LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE)
        sample_input = np.round(sample_input * np.concatenate([letter_embedding_factors] * LETTERS_PER_GROUP)).astype(np.int32)
        factors_for_group = np.concatenate([letter_embedding_factors] * (model.group_inter_linear.in_features // len(letter_embedding_factors)))
        group_inter_linear = quantize_linear(model.group_inter_linear, factors_for_group, sample_input)
    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-group-inter.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(model.lang, GeneratorSharedNet.NO_GROUP_INTER, False).eval().load(REDUCED_MODEL_PATH)
    else:
        reduced_model = GeneratorSharedNet(model, GeneratorSharedNet.NO_GROUP_INTER, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess1)
        train(reduced_model, data, 5, DEFAULT_LEARNING_RATE, epoch_offset=epoch_offset, skip_first_train=True)
        epoch_offset += 5
        reduced_model.save(REDUCED_MODEL_PATH)

    print('Quantize group_output_linear layer')

    def postprocess2(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy()
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_x *= factors_for_group
            group_x = np.round(group_x).astype(np.int32)
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            group_y = group_y.astype(np.float32)
            group_y /= group_output_linear.factors
            y.append(group_y)
        return torch.from_numpy(np.concatenate(y).astype(np.float32))

    reduced_model.eval()
    with torch.no_grad():
        sample_input = convert_sample_input(sample_input, lambda x: relu(group_inter_linear(x)))
        group_output_linear = quantize_linear(reduced_model.group_output_linear, group_inter_linear.factors, sample_input)
    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-group.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(reduced_model.lang, GeneratorSharedNet.NO_GROUP, False).eval().load(REDUCED_MODEL_PATH)
    else:
        reduced_model = GeneratorSharedNet(reduced_model, GeneratorSharedNet.NO_GROUP, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess2)
        train(reduced_model, data, 5, DEFAULT_LEARNING_RATE, epoch_offset=epoch_offset, skip_first_train=True)
        epoch_offset += 5
        reduced_model.save(REDUCED_MODEL_PATH)

    print('Quantize head_inter_linear layer')

    def postprocess3(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy()
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_x *= factors_for_group
            group_x = np.round(group_x).astype(np.int32)
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            y.append(group_y)
        y = np.concatenate(y)
        y = relu(head_inter_linear(y))
        y = y.astype(np.float32)
        y /= head_inter_linear.factors
        return torch.from_numpy(y.astype(np.float32))
    
    def context_to_head_input(x: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            y.append(group_y)
        return np.concatenate(y)

    reduced_model.eval()
    with torch.no_grad():
        sample_input = create_sample_input(dataset_with_letter_embedding)
        del dataset_with_letter_embedding
        sample_input = np.round(sample_input * np.concatenate([letter_embedding_factors] * LETTERS_PER_CONTEXT)).astype(np.int32)
        sample_input = convert_sample_input(sample_input, context_to_head_input)
        head_inter_linear = quantize_linear(reduced_model.head_inter_linear, np.concatenate([group_output_linear.factors] * GROUPS_PER_CONTEXT), sample_input)
    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-head-inter.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(reduced_model.lang, GeneratorSharedNet.NO_HEAD_INTER, False).eval().load(REDUCED_MODEL_PATH)
    else:
        reduced_model = GeneratorSharedNet(reduced_model, GeneratorSharedNet.NO_HEAD_INTER, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess3)
        train(reduced_model, data, 5, DEFAULT_LEARNING_RATE, epoch_offset=epoch_offset, skip_first_train=True)
        epoch_offset += 5
        reduced_model.save(REDUCED_MODEL_PATH)

    print('Quantize head_output_linear layer')

    reduced_model.eval()
    with torch.no_grad():
        sample_input = convert_sample_input(sample_input, lambda x: relu(head_inter_linear(x)))
        head_output_linear = quantize_linear(reduced_model.head_output_linear, head_inter_linear.factors, sample_input)

    head_softmax = quantize_scaled_softmax(3, 4, head_output_linear.factors)

    output_model = {
        'lang': model.lang.CODE,
        'letters': ''.join(model.lang.LETTER_TO_INDEX.keys()),
        'letters_embedding': np.round(letter_to_embedding * letter_embedding_factors).tolist(),
        'group': group_inter_linear.store() + relu.store() + group_output_linear.store() + relu.store(),
        'head': head_inter_linear.store() + relu.store() + head_output_linear.store() + head_softmax.store(),
    }

    return output_model


def punctuation_stats(lang: LangConfig, output_model: dict):
    header('Calculating punctuation statistics')

    def prob_normalize(probabilities, total=9999):
        sum_prob = sum(probabilities)
        probabilities = [x / sum_prob * total for x in probabilities]
        int_prob = [int(round(x)) for x in probabilities]
        while sum(int_prob) > total:
            diff = [probabilities[i] - int_prob[i] for i in range(len(probabilities))]
            min_diff = min(diff)
            index = diff.index(min_diff)
            assert int_prob[index] > 0
            int_prob[index] -= 1
        while sum(int_prob) < total:
            diff = [probabilities[i] - int_prob[i] for i in range(len(probabilities))]
            max_diff = max(diff)
            index = diff.index(max_diff)
            int_prob[index] += 1
        cumulative_sum = [sum(int_prob[:i + 1]) for i in range(len(int_prob))]
        return cumulative_sum

    text = lang.get_text()
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"[\s,.]*\.[\s,.]*", '.', text)
    text = re.sub(r"[\s,]*,[\s,]*", ',', text)
    text = re.sub(r'\s*\w+\s*', 'a', text)

    index = 0
    prob_dot = [0] * (MAX_WORDS_IN_SENTENCE + 1)
    max_words = 0

    while index < len(text):
        next_dot = text.find('.', index)
        if next_dot <= 0:
            break
        words = text[index:next_dot].count('a')
        if words > MAX_WORDS_IN_SENTENCE:
            index = next_dot + 1
            continue
        max_words = max(max_words, words)
        prob_dot[words] += 1
        index = next_dot + 1

    prob_dot[0] = 0
    prob_dot[1] = 0
    cumsum_dot = prob_normalize(prob_dot)

    index = 0

    prob_comma = [[0] * (MAX_WORDS_TO_COMMA + 1) for _ in range(MAX_WORDS_IN_SENTENCE + 1)]

    while index < len(text):
        next_dot = text.find('.', index)
        next_comma = text.find(',', index)
        if next_dot <= 0 or next_comma <= 0:
            break
        words_to_the_end_of_sentence = text[index:next_dot].count('a')
        words_to_comma = text[index:next_comma].count('a')
        if words_to_the_end_of_sentence > MAX_WORDS_IN_SENTENCE or words_to_comma > MAX_WORDS_TO_COMMA:
            index = next_comma + 1
            continue
        if words_to_comma > words_to_the_end_of_sentence:
            words_to_comma = words_to_the_end_of_sentence
        prob_comma[words_to_the_end_of_sentence][words_to_comma] += 1
        index = next_comma + 1

    prob_comma[0][0] = 1

    prob_comma[1][0] = 0
    prob_comma[1][1] = 1

    prob_comma[2][0] = 0
    prob_comma[2][1] = 0
    prob_comma[2][2] = 1

    cumsum_comma = [prob_normalize(prob_comma[i][:i + 1]) for i in range(MAX_WORDS_IN_SENTENCE + 1)]

    output_model['prob_dot'] = cumsum_dot
    output_model['prob_comma'] = cumsum_comma


def train_language(lang_code, model_json_file: Path):
    global epoch_offset
    header(f'Training language: {lang_code}')
    epoch_offset = 0
    lang = get_lang_config(lang_code)
    model = train_basic_model(lang)
    letter_to_embedding, letter_embedding_factors = adjust_letter_embeddings(model)
    model = train_direct_model(model, letter_to_embedding)
    output_model = quantize_model(model, letter_to_embedding, letter_embedding_factors)
    punctuation_stats(lang, output_model)
    model_json_file.parent.mkdir(parents=True, exist_ok=True)
    model_json_file.write_text(json.dumps(output_model))

def generate_files(model_json_file: Path):
    output_model = json.loads(model_json_file.read_text())
    format_c(output_model, model_json_file)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network models for different languages')
    parser.add_argument('languages', nargs='*', help='List of language codes to train (default: all available languages)')
    parser.add_argument('-g', '--gen-only', action='store_true', help='Do not train, only generate output files from previously trained models saved in JSON files.')

    args = parser.parse_args()

    # Determine which languages to process
    lang_list = args.languages if args.languages else get_languages()
    #lang_list = ['la']
    
    for lang_code in lang_list:
        model_json_file = Path(__file__).parent.parent / f"models/{lang_code}.json"
        if not args.gen_only:
            train_language(lang_code, model_json_file)
        #generate_files(model_json_file)


if __name__ == '__main__':
    main()
