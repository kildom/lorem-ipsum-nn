
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
from dataset import TextDataset
from model import GROUPS_PER_CONTEXT, GeneratorSharedNet, LETTER_EMBEDDING_SIZE, LETTER_EMBEDDING_INTER_SIZE, LETTERS_PER_GROUP
from quantizer import QuantizedLinear, QuantizedReLU
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
    def train_callback(model, *_):
        with torch.no_grad():
            emb = model.get_letter_embedding().detach().cpu().numpy()
            max_values = np.max(emb, axis=0)
            print("          - Letter embeddings max values:  ", max_values)
    if BASIC_MODEL_PATH.exists():
        header(f'Loading basic model {BASIC_MODEL_PATH}')
        model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, False)
        model.load_state_dict(torch.load(BASIC_MODEL_PATH))
        return model

    header('Training basic model with leaky ReLU')
    model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, True)
    data = TextDataset(lang)
    train(model, data, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, train_callback, epoch_offset=epoch_offset)
    epoch_offset += DEFAULT_EPOCHS

    header('Re-training basic model with normal ReLU')
    model = GeneratorSharedNet(model, GeneratorSharedNet.ALL, False)
    train(model, data, DEFAULT_EPOCHS // 2, DEFAULT_LEARNING_RATE, train_callback, epoch_offset=epoch_offset)
    epoch_offset += DEFAULT_EPOCHS // 2
    BASIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BASIC_MODEL_PATH)
    return model


def adjust_linear_input(linear_layer: nn.Linear, factors: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    for i in range(linear_layer.out_features):
        for j in range(linear_layer.in_features):
            factor_index = j % factors.shape[0]
            linear_layer.weight[i][j] /= factors[factor_index]


def adjust_letter_embeddings(model: GeneratorSharedNet):
    header('Adjusting letter embeddings')
    LETTER_EMBEDDINGS_FILE = Path(__file__).parent.parent / f"data/{model.lang.CODE}/letter-embeddings.json"

    with torch.no_grad():

        def get_max_letter_embedding():
            emb = model.get_letter_embedding().detach().cpu().numpy()
            return np.max(emb, axis=0)

        # === Adjust weight and bias of linear layers, so letter embeddings are in range [0, 254]

        # Calculate needed adjustment
        max_values = get_max_letter_embedding()
        print("Max values:  ", max_values)
        assert min(max_values) > 0, "Embeddings values should be greater than 0."
        letter_feature_adjustments = 255 / max_values
        print("Adjustments: ", letter_feature_adjustments)
        # Adjust the weight and bias of the letter embedding output layer
        for i in range(LETTER_EMBEDDING_SIZE):
            model.letter_embedding[2].bias[i] *= letter_feature_adjustments[i]
            for j in range(LETTER_EMBEDDING_INTER_SIZE):
                model.letter_embedding[2].weight[i][j] *= letter_feature_adjustments[i]
        # Adjust the weight of the group embedding input layer
        adjust_linear_input(model.group_embedding[0], letter_feature_adjustments)
        # Check the adjustments
        max_values = get_max_letter_embedding()
        print("Adjusted max values:  ", max_values)
        assert min(max_values) > 254.5
        assert max(max_values) < 255.5
        # Print and save final embeddings for each letter
        letter_to_embedding = [[]] * model.lang.ALPHABET_LENGTH
        emb = model.get_letter_embedding().detach().cpu().numpy()
        for i in range(model.lang.ALPHABET_LENGTH):
            vect = [max(0, int(round(float(x)))) for x in emb[i]]
            #print(f"{i:2d} {model.lang.INDEX_TO_LETTER[i]}: {vect}")
            letter_to_embedding[i] = vect

        LETTER_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        LETTER_EMBEDDINGS_FILE.write_text(json.dumps({
            'alphabet': model.lang.ALPHABET,
            'embedding': letter_to_embedding,
        }))

        return letter_to_embedding


def train_direct_model(model, letter_to_embedding):
    global epoch_offset
    DIRECT_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/direct-model.pt"
    if DIRECT_MODEL_PATH.exists():
        header(f'Loading direct model {DIRECT_MODEL_PATH}')
        model = GeneratorSharedNet(model.lang, GeneratorSharedNet.NO_LETTER, False)
        model.load_state_dict(torch.load(DIRECT_MODEL_PATH))
        return model
    header('Training direct model')
    new_model = GeneratorSharedNet(model, GeneratorSharedNet.NO_LETTER, False)
    data = TextDataset(model.lang, letter_to_embedding)
    train(new_model, data, 4, DEFAULT_LEARNING_RATE / 2, epoch_offset=epoch_offset)
    epoch_offset += 4
    DIRECT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_model.state_dict(), DIRECT_MODEL_PATH)
    return new_model


def quantize_model(model: GeneratorSharedNet, letter_to_embedding):
    global epoch_offset

    header('Quantize Model')
    relu = QuantizedReLU()

    print('Quantize group_inter_linear layer')

    def postprocess1(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy().astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x)).astype(np.float32)
            group_y /= group_inter_linear.factors
            y.append(group_y)
        return torch.from_numpy(np.concatenate(y).astype(np.float32))

    model.eval()
    with torch.no_grad():
        group_inter_linear = QuantizedLinear(model.group_inter_linear, np.array([0, 255], dtype=np.int32))
    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-group-inter.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(model, GeneratorSharedNet.NO_GROUP_INTER, False).eval()
        reduced_model.load_state_dict(torch.load(REDUCED_MODEL_PATH))
    else:
        reduced_model = GeneratorSharedNet(model, GeneratorSharedNet.NO_GROUP_INTER, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess1)
        train(reduced_model, data, 2, DEFAULT_LEARNING_RATE / 2, epoch_offset=epoch_offset)
        epoch_offset += 2
        torch.save(reduced_model.state_dict(), REDUCED_MODEL_PATH)

    print('Quantize group_output_linear layer')

    def postprocess2(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy().astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            group_y = group_y.astype(np.float32)
            group_y /= group_output_linear.factors
            y.append(group_y)
        return torch.from_numpy(np.concatenate(y).astype(np.float32))

    model.eval()
    with torch.no_grad():
        adjust_linear_input(reduced_model.group_output_linear, group_inter_linear.factors)
        values_range = group_inter_linear.output_range.copy()
        values_range[:, 0] = 0
        group_output_linear = QuantizedLinear(reduced_model.group_output_linear, values_range)
    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-group.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(reduced_model.lang, GeneratorSharedNet.NO_GROUP, False).eval()
        reduced_model.load_state_dict(torch.load(REDUCED_MODEL_PATH))
    else:
        reduced_model = GeneratorSharedNet(reduced_model, GeneratorSharedNet.NO_GROUP, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess2)
        train(reduced_model, data, 2, DEFAULT_LEARNING_RATE / 2, epoch_offset=epoch_offset)
        epoch_offset += 2
        torch.save(reduced_model.state_dict(), REDUCED_MODEL_PATH)

    print('Quantize head_inter_linear layer')

    def postprocess3(xx):
        x: 'npt.NDArray[np.float32]' = xx.detach().numpy().astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            y.append(group_y)
        y = np.concatenate(y)
        y = relu(head_inter_linear(y)).astype(np.float32)
        y /= head_inter_linear.factors
        return torch.from_numpy(y.astype(np.float32))

    model.eval()
    with torch.no_grad():
        adjust_linear_input(reduced_model.head_inter_linear, group_output_linear.factors)
        values_range = np.concatenate([group_output_linear.output_range.copy()] * GROUPS_PER_CONTEXT)
        values_range[:, 0] = 0
        head_inter_linear = QuantizedLinear(reduced_model.head_inter_linear, values_range)

    REDUCED_MODEL_PATH = Path(__file__).parent.parent / f"data/{model.lang.CODE}/reduced-model-no-head-inter.pt"
    if REDUCED_MODEL_PATH.exists():
        reduced_model = GeneratorSharedNet(reduced_model.lang, GeneratorSharedNet.NO_HEAD_INTER, False).eval()
        reduced_model.load_state_dict(torch.load(REDUCED_MODEL_PATH))
    else:
        reduced_model = GeneratorSharedNet(reduced_model, GeneratorSharedNet.NO_HEAD_INTER, False).eval()
        data = TextDataset(model.lang, letter_to_embedding, postprocess3)
        train(reduced_model, data, 2, DEFAULT_LEARNING_RATE / 2, epoch_offset=epoch_offset)
        epoch_offset += 2
        torch.save(reduced_model.state_dict(), REDUCED_MODEL_PATH)

    print('Quantize head_output_linear layer')

    model.eval()
    with torch.no_grad():
        adjust_linear_input(reduced_model.head_output_linear, head_inter_linear.factors)
        values_range = head_inter_linear.output_range.copy()
        values_range[:, 0] = 0
        head_output_linear = QuantizedLinear(reduced_model.head_output_linear, values_range)

    print('Quantize output postprocessing')

    prob_frac_bits = 32
    invalid = True
    while invalid:
        invalid = False
        prob_frac_bits -= 1
        prob_coef = []
        eval_value = (1 << prob_frac_bits)
        div = 1
        while True:
            group_coef = []
            for j in range(8):
                v = math.floor(math.exp(j / div) * (1 << prob_frac_bits))
                group_coef.append(v)
            if group_coef == [group_coef[0]] * len(group_coef):
                break
            prob_coef.append(group_coef)
            div *= 8
            eval_mul = eval_value * group_coef[-1]
            invalid = invalid or eval_mul > ((1 << 31) - 1)
            eval_value = eval_mul >> prob_frac_bits
    pprint(prob_coef)

    starting_bit_shift = 31 - (len(prob_coef) - 1) * 3

    assert min(head_output_linear.factors) > 1.0001, 'Not implemented for factors less than 1'
    inv_factors_u0_31 = np.round(2147483648 / head_output_linear.factors).astype(np.int32)
    assert min(inv_factors_u0_31) > 10, 'Not implemented for such large factors'
    max_abs_output = int(np.max(np.abs(head_output_linear.output_range)))
    assert max_abs_output < 2147483648, 'Output range is too large for fixed-point s32.0'
    for bit_shift in range(starting_bit_shift, 32):
        min_value = int(np.min(head_output_linear.output_range[:,0].astype(np.int64) * inv_factors_u0_31.astype(np.int64))) >> bit_shift
        max_value = int(np.max(head_output_linear.output_range[:,1].astype(np.int64) * inv_factors_u0_31.astype(np.int64))) >> bit_shift
        max_abs = max(abs(min_value), abs(max_value))
        if ((max_abs * 127) >> 4) < 2147483648:
            break
    fractional_bits = 31 - bit_shift
    print(bit_shift, min_value, max_value, max_abs, fractional_bits)

    def str_to_emb(s: str) -> npt.NDArray[np.int32]:
        r = []
        for c in s:
            index = model.lang.LETTER_TO_INDEX[c]
            r += letter_to_embedding[index]
        return np.array(r, dtype=np.int32)

    heat = 50
    heat_inv_u11_4 = max(2, min(127, 1600 // heat))
    limit_logits = (8 << fractional_bits) - 1

    group0 = relu(group_output_linear(relu(group_inter_linear(str_to_emb('lore')))))
    group1 = relu(group_output_linear(relu(group_inter_linear(str_to_emb('m ip')))))
    group2 = relu(group_output_linear(relu(group_inter_linear(str_to_emb('etaa')))))
    logits_unscaled = head_output_linear(relu(head_inter_linear(np.concatenate([group0, group1, group2]))))
    logits_no_heat_fixed = (logits_unscaled.astype(np.int64) * inv_factors_u0_31.astype(np.int64)) >> bit_shift
    logits_fixed = (logits_no_heat_fixed * heat_inv_u11_4) >> 4
    logits_fixed = logits_fixed - max(logits_fixed) + limit_logits
    logits_fixed = np.maximum(logits_fixed, -2147483648).astype(np.int32)

    def exp_fixed(x):
        if (x < 0):
            return 0
        assert x < (8 << fractional_bits)
        shift = fractional_bits
        result = prob_coef[0][0]
        for group_coef in prob_coef:
            group_value = (x >> shift) & 7
            result = (result * group_coef[group_value]) >> prob_frac_bits
            shift -= 3
            if shift < 0:
                break
        return result >> prob_frac_bits

    exponents_fixed = np.array([exp_fixed(x) for x in logits_fixed], dtype=np.int32)
    prob_cumsum = np.cumsum(exponents_fixed)

    for i in [0, 1, 2, 388, prob_cumsum[-1] - 2, prob_cumsum[-1] - 1]:
        letter_index = random_probs(prob_cumsum, i)
        print(f"{model.lang.INDEX_TO_LETTER[letter_index]}: {exponents_fixed[letter_index]}")

    output_model = {
        'lang': model.lang.CODE,
        'letters': ''.join(model.lang.LETTER_TO_INDEX.keys()),
        'letters_embedding': letter_to_embedding,
        'group': group_inter_linear.store() + relu.store() + group_output_linear.store() + relu.store(),
        'head': head_inter_linear.store() + relu.store() + head_output_linear.store(),
        'output_scale_u0_31': inv_factors_u0_31.tolist(),
        'output_shift': bit_shift,
        'fractional_bits': fractional_bits,
        'prob_fractional_bits': prob_frac_bits,
        'prob_exp_table': prob_coef,
        'empty_group_embedding': relu(group_output_linear(relu(group_inter_linear(str_to_emb('    '))))).tolist(),
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
    letter_to_embedding = adjust_letter_embeddings(model)
    model = train_direct_model(model, letter_to_embedding)
    output_model = quantize_model(model, letter_to_embedding)
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
    
    for lang_code in lang_list:
        model_json_file = Path(__file__).parent.parent / f"models/{lang_code}.json"
        if not args.gen_only:
            train_language(lang_code, model_json_file)
        generate_files(model_json_file)


if __name__ == '__main__':
    main()
