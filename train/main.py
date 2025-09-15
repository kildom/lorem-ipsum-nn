
import socket
import subprocess
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
from model import GROUP_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_SIZE, GROUPS_PER_CONTEXT, HEAD_INTER_SIZE, LETTERS_PER_CONTEXT, GeneratorSharedNet, LETTER_EMBEDDING_SIZE, LETTER_EMBEDDING_INTER_SIZE, LETTERS_PER_GROUP
from quantizer import QuantizedReLU, quantize_linear, quantize_scaled_softmax
from train import train, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from lang import LangConfig, get_lang_config, get_languages
from format_c import format_c
from format_ts import format_ts

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
            max_values = np.max(emb, axis=0).tolist()
            num_zeros = np.sum((emb == 0).astype(np.int32), axis=0).tolist()
            print("          - Letter embeddings max values:  ", max_values, "  zeros:  ", num_zeros, "of", emb.shape[0])

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
        'name': model.lang.NAME,
        'letters': ''.join(model.lang.LETTER_TO_INDEX.keys()),
        'letters_embedding': np.round(letter_to_embedding * letter_embedding_factors).tolist(),
        'group': group_inter_linear.store() + relu.store(GROUP_EMBEDDING_INTER_SIZE) + group_output_linear.store() + relu.store(GROUP_EMBEDDING_SIZE),
        'head': head_inter_linear.store() + relu.store(HEAD_INTER_SIZE) + head_output_linear.store() + head_softmax.store(),
    }

    return output_model


def punctuation_stats(lang: LangConfig, output_model: dict):
    header('Calculating punctuation statistics')

    text = lang.get_text()
    text = text[1:4000000]
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"[\s,.]*\.[\s,.]*", '.', text)
    text = re.sub(r"[\s,]*,[\s,]*", ',', text)
    text = re.sub(r'\s*\w+\s*', 'a', text)
    text = re.sub(r'(?:^|\.)(?:a,?){41,}', '', text)
    text = re.sub(r'\.[^.]*a{21}[a,]*', '', text)
    if text[-1] != '.':
        text += '.'
    text = re.sub(r'\.a(?=\.)', '', text)

    hit_dot = np.zeros((41, 21), dtype=np.int64)
    hit_comma = np.zeros(hit_dot.shape, dtype=np.int64)
    total = np.zeros(hit_dot.shape, dtype=np.int64)

    words_since_dot = 0
    words_since_comma = 0
    for i in range(1, len(text)):
        c = text[i]
        if c == '.':
            hit_dot[words_since_dot][words_since_comma] += 1
            words_since_dot = 0
            words_since_comma = 0
        elif c == ',':
            hit_comma[words_since_dot][words_since_comma] += 1
            words_since_comma = 0
        else:
            words_since_dot += 1
            words_since_comma += 1
            assert words_since_dot < 41 and words_since_comma < 21, f"words since dot: {words_since_dot}, words since comma: {words_since_comma}, position: {i}, text: {text[max(0, i - 50):i]}|{text[i:i + 50]}"
            total[words_since_dot][words_since_comma] += 1

    hit_dot = hit_dot[1:, 1:]
    hit_comma = hit_comma[1:, 1:]
    total = total[1:, 1:]

    # If some case does not occur, assume that there is a dot (for safety)
    zeros = (total == 0).astype(np.int64)
    hit_dot = hit_dot * (1 - zeros) + zeros
    total = total + zeros
    # Export probabilities as uint8
    prob_dot = np.round(np.maximum(0, hit_dot.astype(np.float64) / total.astype(np.float64) * 256 - 1)).astype(np.int32)
    # Limit number of allowed words by setting 100% probability for edge of the array
    prob_dot[-1] = 255

    # If in some case there is a dot for sure, then assume 100% probability of comma just in case
    dot_for_sure = (hit_dot >= total).astype(np.int64)
    # Export probabilities as uint8
    prob_comma = np.round(np.maximum(0, np.maximum(dot_for_sure, hit_comma).astype(np.float64) / np.maximum(dot_for_sure, total - hit_dot).astype(np.float64) * 256 - 1)).astype(np.int32)
    # Limit number of allowed words by setting 100% probability for edges of the array
    prob_comma[-1] = 255
    prob_comma[:,-1] = 255

    prob_dot = prob_dot.tolist()
    prob_comma = prob_comma.tolist()
    for i in range(len(prob_dot)):
        prob_dot[i] = prob_dot[i][:i + 1]
        prob_comma[i] = prob_comma[i][:i + 1]

    output_model['prob_dot'] = prob_dot
    output_model['prob_comma'] = prob_comma


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
    format_ts(output_model, model_json_file)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_tensorboard_if_not_running(port: int = 6006):
    if is_port_in_use(port):
        print(f"TensorBoard is already running on http://localhost:{port}")
    else:
        logdir = (Path(__file__).parent / "../runs").resolve()
        print(f"Starting TensorBoard at http://localhost:{port}")
        subprocess.Popen(
            ["tensorboard", f"--logdir={str(logdir)}", f"--port={port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network models for different languages')
    parser.add_argument('languages', nargs='*', help='List of language codes to train (default: all available languages)')
    parser.add_argument('-g', '--gen-only', action='store_true', help='Do not train, only generate output files from previously trained models saved in JSON files.')

    args = parser.parse_args()

    start_tensorboard_if_not_running()

    # Determine which languages to process
    lang_list = args.languages if args.languages else get_languages()
    #lang_list = ['la']
    
    for lang_code in lang_list:
        model_json_file = Path(__file__).parent.parent / f"models/{lang_code}.json"
        if not args.gen_only:
            train_language(lang_code, model_json_file)
        generate_files(model_json_file)


if __name__ == '__main__':
    main()
