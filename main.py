
import json
import math
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from data_loader import LatinDataset, random_probs
from model import GROUPS_PER_CONTEXT, LETTER_TO_INDEX, LatinSharedNet, LETTER_EMBEDDING_SIZE, ALPHABET_LENGTH, LETTER_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_INTER_SIZE, LETTERS_PER_GROUP, INDEX_TO_LETTER
from quantizer import QuantizedLinear, QuantizedReLU
from train import train, DEFAULT_EPOCHS

def header(text):
    print()
    print('#' * 79)
    print('#' + ' ' * 19 + text)
    print('#' * 79)
    print()


def train_basic_model():
    def train_callback(model, epoch, *_):
        with torch.no_grad():
            emb = model.get_letter_embedding().detach().cpu().numpy()
            max_values = np.max(emb, axis=0)
            print("Max values:  ", max_values)
    if Path("data/basic-model.pt").exists():
        header('Loading basic model')
        model = LatinSharedNet(layers_conf=LatinSharedNet.ALL)
        model.load_state_dict(torch.load("data/basic-model.pt"))
        return model
    header('Training basic model')
    model = LatinSharedNet(layers_conf=LatinSharedNet.ALL)
    data = LatinDataset()
    train(model, data, DEFAULT_EPOCHS, train_callback)
    torch.save(model.state_dict(), "data/basic-model.pt")
    return model


def adjust_linear_input(linear_layer: nn.Linear, factors: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    for i in range(linear_layer.out_features):
        for j in range(linear_layer.in_features):
            factor_index = j % factors.shape[0]
            linear_layer.weight[i][j] /= factors[factor_index]

def adjust_letter_embeddings(model):
    header('Adjusting letter embeddings')

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
        letter_to_embedding = [[]] * ALPHABET_LENGTH
        emb = model.get_letter_embedding().detach().cpu().numpy()
        pprint(emb)
        for i in range(ALPHABET_LENGTH):
            print(f"{i:2d} {INDEX_TO_LETTER[i]}: {[int(round(float(x))) for x in emb[i]]}")
            letter_to_embedding[i] = [int(round(float(x))) for x in emb[i]]
        with open("data/letter-embeddings.json", 'w') as file:
            json.dump(letter_to_embedding, file, indent=2)

        return letter_to_embedding


def train_direct_model(model, letter_to_embedding):
    if Path("data/direct-model.pt").exists():
        header('Loading direct model')
        model = LatinSharedNet(layers_conf=LatinSharedNet.NO_LETTER)
        model.load_state_dict(torch.load("data/direct-model.pt"))
        return model
    header('Training direct model')
    new_model = LatinSharedNet(layers_conf=LatinSharedNet.NO_LETTER, source_model=model)
    data = LatinDataset(letter_to_embedding)
    train(new_model, data, 4)
    torch.save(new_model.state_dict(), "data/direct-model.pt")
    return new_model


def quantize_model(model: LatinSharedNet, letter_to_embedding):

    header('Quantize Model')
    relu = QuantizedReLU()

    print('Quantize group_inter_linear layer')
    def postprocess1(x: npt.NDArray[np.float32]):
        x = x.astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x)).astype(np.float32)
            group_y /= group_inter_linear.factors
            y.append(group_y)
        return np.concatenate(y)
    model.eval()
    with torch.no_grad():
        group_inter_linear = QuantizedLinear(model.group_inter_linear, np.array([0, 255], dtype=np.int32))
        reduced_model = LatinSharedNet(layers_conf=LatinSharedNet.NO_GROUP_INTER, source_model=model).eval()
    data = LatinDataset(letter_to_embedding, postprocess1)
    train(reduced_model, data, 2)

    print('Quantize group_output_linear layer')
    def postprocess2(x: npt.NDArray[np.float32]):
        x = x.astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            group_y = group_y.astype(np.float32)
            group_y /= group_output_linear.factors
            y.append(group_y)
        return np.concatenate(y)

    model.eval()
    with torch.no_grad():
        adjust_linear_input(reduced_model.group_output_linear, group_inter_linear.factors)
        values_range = group_inter_linear.output_range.copy()
        values_range[:, 0] = 0
        group_output_linear = QuantizedLinear(reduced_model.group_output_linear, values_range)
        reduced_model = LatinSharedNet(layers_conf=LatinSharedNet.NO_GROUP, source_model=reduced_model).eval()
    data = LatinDataset(letter_to_embedding, postprocess2)
    train(reduced_model, data, 2)

    print('Quantize head_inter_linear layer')
    def postprocess3(x: npt.NDArray[np.float32]):
        x = x.astype(np.int32)
        y = []
        for i in range(GROUPS_PER_CONTEXT):
            group_x = x[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]
            group_y = relu(group_inter_linear(group_x))
            group_y = relu(group_output_linear(group_y))
            y.append(group_y)
        y = np.concatenate(y)
        y = relu(head_inter_linear(y)).astype(np.float32)
        y /= head_inter_linear.factors
        return y
    model.eval()
    with torch.no_grad():
        adjust_linear_input(reduced_model.head_inter_linear, group_output_linear.factors)
        values_range = np.concatenate([group_output_linear.output_range.copy()] * GROUPS_PER_CONTEXT)
        values_range[:, 0] = 0
        head_inter_linear = QuantizedLinear(reduced_model.head_inter_linear, values_range)
        reduced_model = LatinSharedNet(layers_conf=LatinSharedNet.NO_HEAD_INTER, source_model=reduced_model).eval()
    data = LatinDataset(letter_to_embedding, postprocess3)
    train(reduced_model, data, 2)

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
            index = LETTER_TO_INDEX[c]
            r += letter_to_embedding[index]
        return np.array(r, dtype=np.int32)

    def softmax(x):
        e_x = np.exp(x)  # for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

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
        print(f"{INDEX_TO_LETTER[letter_index]}: {exponents_fixed[letter_index]}")

    output_dict = {
        'letters': ''.join(LETTER_TO_INDEX.keys()),
        'letters_embedding': letter_to_embedding,
        'group': group_inter_linear.store() + relu.store() + group_output_linear.store() + relu.store(),
        'head': head_inter_linear.store() + relu.store() + head_output_linear.store(),
        'output_scale_u0_31': inv_factors_u0_31.tolist(),
        'output_shift': bit_shift,
        'fractional_bits': fractional_bits,
        'prob_fractional_bits': prob_frac_bits,
        'prob_exp_table': prob_coef,
        'empty_group_embedding': relu(group_output_linear(relu(group_inter_linear(str_to_emb('    '))))),
    }

    with open('data/quantized-model.json', 'w') as f:
        json.dump(output_dict, f, indent=2)

    # logits = logits_fixed.astype(np.float32) / (1 << fractional_bits)
    # probabilities = softmax(logits)

    #pprint(([f'{INDEX_TO_LETTER[i]}: {x >> prob_frac_bits}' for i, x in enumerate(exponents_fixed)], np.exp(logits).astype(np.int32)))
    pprint((exponents_fixed, prob_cumsum))

    #pprint((logits, logits_fixed, (probabilities * 65536).astype(np.int32), sum(probabilities)))

    #for i in range(19):
     #   pprint((i, 33554432 * math.exp(4 / (1 << i)), math.exp(4 / (1 << i))))

    # pprint(head_output_linear.output_range[:,1])
    # pprint(head_output_linear.factors)
    # pprint(inv_factors_u0_31)
    # pprint(head_output_linear.output_range[:,0] / head_output_linear.factors)
    # pprint(head_output_linear.output_range[:,1] / head_output_linear.factors)
    # pprint(max_abs_output)

model = train_basic_model()
letter_to_embedding = adjust_letter_embeddings(model)
model = train_direct_model(model, letter_to_embedding)
quantize_model(model, letter_to_embedding)
