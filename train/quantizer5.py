import math
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import random
from pathlib import Path
from lang import get_lang_config
from model import GeneratorSharedNet
from pprint import pprint
import torch.nn.functional as F


class SimpleQuantizedModel:
    def __call__(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        return self.forward(x)

    def store(self) -> 'list[dict]':
        raise NotImplementedError("Subclasses must implement the store method to save the model state.")


class QuantizedLinear(SimpleQuantizedModel):

    def __init__(self,
                 weight: 'npt.NDArray[np.int32]',
                 bias: 'npt.NDArray[np.int32]',
                 input_shift: 'npt.NDArray[np.int32]',
                 input_clamp: 'npt.NDArray[np.int32]',
                 factors: 'npt.NDArray[np.float64]'=None,
                 ):
        self.weight = weight
        self.bias = bias
        self.input_shift = input_shift
        self.input_clamp = input_clamp
        self.factors = factors

    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        x = x >> self.input_shift
        x = np.clip(x, self.input_clamp[0], self.input_clamp[1])
        y = self.bias + np.dot(self.weight, x)
        return y

    def store(self) -> 'list[dict]':
        result = []
        if np.max(self.input_shifts) > 0:
            result.append({
                'type': 'bit_shift',
                'value': self.input_shifts.tolist()
            })
        result.append({
            'type': 'linear',
            'weight': self.weight.tolist(),
            'bias': self.bias.tolist()
        })
        return result

    @staticmethod
    def load(layers: list[dict]) -> 'SimpleQuantizedModel|None':
        if len(layers) >= 2 and layers[0]['type'] == 'bit_shift' and layers[1]['type'] == 'linear':
            input_shifts = np.array(layers[0]['value'], dtype=np.int32)
            layers.pop(0)
        else:
            input_shifts = None
        if layers[0]['type'] != 'linear':
            return None
        weight = np.array(layers[0]['weight'], dtype=np.int32)
        bias = np.array(layers[0]['bias'], dtype=np.int32)
        layers.pop(0)
        if input_shifts is None:
            input_shifts = np.zeros(weight.shape[1], dtype=np.int32)
        return QuantizedLinear(weight, bias, input_shifts)


class QuantizedReLU(SimpleQuantizedModel):

    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        return np.maximum(0, x)

    def store(self) -> 'list[dict]':
        return [{
            'type': 'relu'
        }]

    @staticmethod
    def load(layers: list[dict]) -> 'SimpleQuantizedModel|None':
        if layers[0]['type'] != 'relu':
            return None
        layers.pop(0)
        return QuantizedReLU()

def analyze_sample_input(sample_input: 'npt.NDArray[np.int32]'):
    min_input = np.min(sample_input, axis=0)
    max_input = np.max(sample_input, axis=0)
    max_abs = np.maximum(-min_input, max_input)
    half_range = max_abs / 2
    v = np.maximum(np.abs(sample_input), half_range)
    estimated_input = np.ceil(np.average(v, axis=0)).astype(np.int32)
    return min_input, max_input, estimated_input

def find_biggest_addends(addends: 'npt.NDArray[np.int64]') -> set[int]:
    max_addends_indexes = np.argmax(addends, axis=1)
    output = np.sum(addends, axis=1)
    valid_output = (output <= 1073741823).astype(np.int64)
    max_addends_invalid_indexes = max_addends_indexes * (1 - valid_output) - valid_output
    result = set([int(x) for x in np.unique(max_addends_invalid_indexes) if x >= 0])
    #pprint((addends, max_addends_indexes, max_addends_invalid_indexes, result))
    return result

def check_input_clamp(weight: 'npt.NDArray[np.int32]', min_input: 'npt.NDArray[np.int64]', max_input: 'npt.NDArray[np.int64]') -> bool:
    if np.max(np.abs(min_input)) > 1073741823 or np.max(np.abs(max_input)) > 1073741823:
        return False
    positive_weight = np.maximum(0, weight).astype(np.int64)
    negative_weight = np.minimum(0, weight).astype(np.int64)
    positive_mod = int(np.max(np.abs(np.dot(positive_weight, max_input) + np.dot(negative_weight, min_input))))
    negative_mod = int(np.max(np.abs(np.dot(positive_weight, min_input) + np.dot(negative_weight, max_input))))
    #pprint((positive_mod, negative_mod, min_input.dtype))
    return max(positive_mod, negative_mod) <= 1073741823

def quantize_linear(linear: nn.Linear, input_factors: 'npt.NDArray[np.float64]', sample_input: 'npt.NDArray[np.int32]') -> QuantizedLinear:
    initial_weight = linear.weight.detach().cpu().numpy().astype(np.float64)
    initial_bias = linear.bias.detach().cpu().numpy().astype(np.float64)
    initial_min_input, initial_max_input, estimated_input = analyze_sample_input(sample_input)
    input_shift = np.zeros(len(input_factors), dtype=np.int32)
    while True:
        min_input = initial_min_input >> input_shift
        max_input = initial_max_input >> input_shift
        # Adjust weights to take into account input factors and input bit shift
        weight = initial_weight * (1 << input_shift) / input_factors
        # Maximize at least one weight to 127 in each row by calculating output factors
        max_weight = np.max(np.abs(weight), axis=1)
        max_max_weight  = np.max(max_weight)
        max_weight = np.maximum(max_weight, max_max_weight / 1000000)
        output_factors = 127 / max_weight
        weight = weight * output_factors.reshape(-1, 1)
        # If possible, maximize weights in each column by shifting the input
        max_weight = np.max(np.abs(weight), axis=0)
        can_be_updated = (((estimated_input >> input_shift) / max_weight > 6) & (max_weight <= 63)).astype(np.int32)
        if max(can_be_updated) > 0:
            input_shift += can_be_updated
            continue
        # Calculate final quantized weight
        weight = np.round(weight)
        assert np.max(np.abs(weight)) <= 127
        weight = weight.astype(np.int32)
        # Check if maximum values during dot operation are within limits of int31
        positive_weight = np.maximum(0, weight).astype(np.int64)
        negative_weight = np.minimum(0, weight).astype(np.int64)
        positive_addends = np.abs(positive_weight * max_input) + np.abs(negative_weight * min_input)
        negative_addends = np.abs(positive_weight * min_input) + np.abs(negative_weight * max_input)
        if np.max(np.sum(positive_addends, axis=1)) > 1073741823 or np.max(np.sum(negative_addends, axis=1)) > 1073741823:
            indexes = find_biggest_addends(positive_addends)
            indexes = indexes.union(find_biggest_addends(negative_addends))
            #pprint(('bitshift', indexes, input_shift))
            for i in indexes:
                input_shift[i] += 1
            #pprint(('bitshift', indexes, input_shift))
            continue
        # Calculate bias
        bias = np.round(initial_bias * output_factors).astype(np.int64)
        if np.max(np.abs(bias)) > 1073741823:
            input_shift += 1
            continue
        bias = bias.astype(np.int32)
        #pprint((weight, initial_weight))
        break
    #print(weight, bias, input_size, output_size)
    # Make clamping wider if possible
    increment = 1000 * (max_input - min_input).astype(np.int64)
    increment = np.maximum(increment, np.max(increment) // 100)
    while np.max(increment) > 1:
        if check_input_clamp(weight, min_input - increment, max_input + increment):
            min_input = (min_input - increment).astype(np.int32)
            max_input = (max_input + increment).astype(np.int32)
        increment = increment >> 1
    #pprint((increment, min_input, max_input))
    input_clamp = np.stack((min_input, max_input), axis=0)
    return QuantizedLinear(weight, bias, input_shift, input_clamp, output_factors)

EXP_TABLE = [
    [512, 1392, 3783, 10284, 27954, 75988, 206556, 561476],
    [512, 580, 657, 745, 844, 957, 1084, 1228],
    [512, 520, 528, 537, 545, 554, 562, 571],
    [512, 513, 514, 515, 516, 517, 518, 519]
]

def create_exp_table():
    def exp_int(x, table):
        result = table[0][0]
        for k in range(4):
            chunk = (x >> (9 - 3 * k)) & 7
            result *= table[k][chunk]
            result >>= 9
        return result
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
                v = round(math.exp(j / div) * (1 << prob_frac_bits))
                group_coef.append(v)
            if group_coef == [group_coef[0]] * len(group_coef):
                break
            prob_coef.append(group_coef)
            div *= 8
            eval_mul = eval_value * group_coef[-1]
            invalid = invalid or eval_mul > ((1 << 31) - 1)
            eval_value = eval_mul >> prob_frac_bits
    res = np.zeros((8 << 9, 2))
    for i in range(8 << 9):
        value1 = exp_int(i, EXP_TABLE) / (1 << prob_frac_bits)
        value2 = exp_int(i, prob_coef) / (1 << prob_frac_bits)
        value3 = math.exp(i / (1 << 9))
        print(value1 - value3, value2 - value3, value3)
        res[i][0] = value1 - value3
        res[i][1] = value2 - value3
    print(np.min(res, axis=0), np.max(res, axis=0))
    pprint(prob_coef)
    exit()
#create_exp_table()


class QuantizedScaledSoftmax(SimpleQuantizedModel):

    def __init__(self,
                 weight: 'npt.NDArray[np.int32]',
                 frac_bits: int,
                 ):
        self.weight = weight
        self.frac_bits = frac_bits
        self.scale = 0
        self.range_top = (8 << frac_bits) - 1
        self.factors = np.full(len(weight), 1522380, dtype=np.float64)

    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        weight_scaled = self.weight * self.scale
        x = x.astype(np.int64) * weight_scaled.astype(np.int64)
        # if self.frac_bits > 32 + 10: # We are using only 9 bits for fractional part plus 1 bit for margin during range shifting.
        # On platforms where it makes sense, lower 32 bits can be dropped and we use smaller fractional part.
        x = x - np.max(x) + self.range_top
        x = (x >> (self.frac_bits - 9)).astype(np.int32) # We use only 9 bits for fractional part during exponentiation.
        for i in range(len(x)):
            value = int(x[i])
            if value >= 0:
                result = EXP_TABLE[0][0]
                for k in range(4):
                    chunk = (value >> (9 - 3 * k)) & 7
                    result *= EXP_TABLE[k][chunk]
                    result >>= 9
                x[i] = result
            else:
                x[i] = 0
        return x >> 9

def quantize_scaled_softmax(scale_int_bits: int, scale_frac_bits: int, input_factors: 'npt.NDArray[np.float64]') -> QuantizedScaledSoftmax:
    input_bits = 32
    scale_bits = scale_int_bits + scale_frac_bits
    assert np.max(input_factors) >= 1, f"Input factors {input_factors} must be greater than or equal to 1."
    assert scale_bits <= 16, f"Integer bits {scale_int_bits} and fractional bits {scale_frac_bits} must be less than or equal to 16."
    max_weight_bits = min(62 - input_bits - scale_bits, 30 - scale_bits) # Maximum is 2 bits lower than integer, one bit for sign, one bit of margin needed when moving entire range to fixed top value
    max_weight = (1 << max_weight_bits) - 1
    weight_float = 1 / input_factors
    weight_float_max = float(np.max(np.abs(weight_float)))
    weight_frac_bits = 0
    while np.round(weight_float_max * (1 << (weight_frac_bits + 1))).astype(np.int64) <= max_weight:
        weight_frac_bits += 1
    weight_fixed = np.round(weight_float * (1 << weight_frac_bits)).astype(np.int32)
    #pprint((weight_float, weight_fixed, weight_frac_bits + scale_frac_bits))
    result = QuantizedScaledSoftmax(weight_fixed, weight_frac_bits + scale_frac_bits)
    return result

########################################################################################################################
########################################################################################################################
########################################################################################################################


def test():
    np.set_printoptions(
        linewidth=150,  # wider output
        threshold=np.inf,  # print entire array, no summarizing
        precision=4,  # 4 decimal places
        suppress=True  # don't use scientific notation for small values
    )

    lang_code = 'la'
    lang = get_lang_config(lang_code)
    MODEL_PATH = Path(__file__).parent.parent / f'data/{lang_code}/basic-model.pt'
    model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, False).eval()
    model.load_state_dict(torch.load(MODEL_PATH))

    letter_embedding_precise = model.get_letter_embedding().detach().cpu().numpy()
    max_letter_embedding = np.max(letter_embedding_precise, axis=0)
    letter_factors = 255 / max_letter_embedding
    letter_embedding_float = np.round(letter_embedding_precise * letter_factors) / letter_factors
    letter_embedding = np.round(letter_embedding_float * letter_factors).astype(np.int32)

    letters_per_group = model.group_inter_linear.in_features // len(letter_embedding[0])
    groups_per_context = model.head_inter_linear.in_features // model.group_output_linear.out_features
    relu = QuantizedReLU()


    # Quantize group intermediate layer
    group_inter_linear = quantize_linear(
        model.group_inter_linear,
        np.concatenate([letter_factors] * letters_per_group),
        np.concatenate([letter_embedding] * letters_per_group, axis=1)
        )

    layer1_outputs = np.zeros((10000, model.group_inter_linear.out_features), dtype=np.int32)
    for i in range(layer1_outputs.shape[0]):
        x = np.concatenate([letter_embedding[int(k) % lang.ALPHABET_LENGTH] for k in random.randbytes(letters_per_group)])
        layer1_outputs[i] = relu(group_inter_linear(x))

    # Quantize group output layer
    group_output_linear = quantize_linear(
        model.group_output_linear,
        group_inter_linear.factors,
        layer1_outputs
        )
    layer2_outputs = np.zeros((len(layer1_outputs), model.group_output_linear.out_features), dtype=np.int32)
    for i in range(layer1_outputs.shape[0]):
        layer2_outputs[i] = relu(group_output_linear(layer1_outputs[i]))

    # Quantize head intermediate layer
    layer3_inputs = layer2_outputs[:layer2_outputs.shape[0] // groups_per_context * groups_per_context].reshape(-1, layer2_outputs.shape[1] * groups_per_context)
    head_inter_linear = quantize_linear(
        model.head_inter_linear,
        np.concatenate([group_output_linear.factors] * groups_per_context),
        layer3_inputs
        )
    layer3_outputs = np.zeros((layer3_inputs.shape[0], model.head_inter_linear.out_features), dtype=np.int32)
    for i in range(layer3_outputs.shape[0]):
        layer3_outputs[i] = relu(head_inter_linear(layer3_inputs[i]))
    #pprint(('head_inter_linear estimated output', analyze_sample_input(layer3_outputs)))

    # Quantize head output layer
    head_output_linear = quantize_linear(
        model.head_output_linear,
        head_inter_linear.factors,
        layer3_outputs
        )
    layer4_outputs = np.zeros((len(layer3_outputs), model.head_output_linear.out_features), dtype=np.int32)
    for i in range(layer3_outputs.shape[0]):
        layer4_outputs[i] = relu(head_output_linear(layer3_outputs[i]))
    # pprint(('head_output_linear estimated output', analyze_sample_input(layer4_outputs)))
    # pprint(('group_inter_linear', group_inter_linear.weight, group_inter_linear.bias, group_inter_linear.input_shift, group_inter_linear.input_clamp))
    # pprint(('group_output_linear', group_output_linear.weight, group_output_linear.bias, group_output_linear.input_shift, group_output_linear.input_clamp))
    # pprint(('head_inter_linear', head_inter_linear.weight, head_inter_linear.bias, head_inter_linear.input_shift, head_inter_linear.input_clamp))
    # pprint(('head_output_linear', head_output_linear.weight, head_output_linear.bias, head_output_linear.input_shift, head_output_linear.input_clamp))

    softmax = quantize_scaled_softmax(3, 4, head_output_linear.factors)

    def next_char(text: str, use_float) -> 'tuple[str, int]':
        input1 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[0]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[1]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[2]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[3]]]),
        ])
        input2 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[4]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[5]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[6]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[7]]]),
        ])
        input3 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[8]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[9]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[10]]]),
            (letter_embedding[lang.LETTER_TO_INDEX[text[11]]]),
        ])

        def group(input: 'npt.NDArray[np.int32]') -> 'tuple[npt.NDArray[np.int32], int]':
            x = input
            x = group_inter_linear(x)
            x = relu(x)
            x = group_output_linear(x)
            x = relu(x)
            return x

        x1 = group(input1)
        x2 = group(input2)
        x3 = group(input3)

        x = np.concatenate([x1, x2, x3])
        x = head_inter_linear(x)
        x = relu(x)
        tr_head_inter = x
        x = head_output_linear(x)

        #assert sh == 0, f'Head shift {sh} is not zero.'

        heat = 0.6
        softmax.scale = int(16 / heat)

        x_fixed = x
        x = x.astype(np.float64) / head_output_linear.factors
        if use_float:
            with torch.no_grad():
                xx = np.zeros(12 * lang.ALPHABET_LENGTH, dtype=np.float32)
                for i, c in enumerate(text):
                    index = lang.LETTER_TO_INDEX[c]
                    xx[i * lang.ALPHABET_LENGTH + index] = 1.0
                xx = torch.from_numpy(xx)
                model.eval()
                yy = model.forward_with_tracking(xx)
                yy = yy.detach().cpu().numpy()
            with open('tmp.csv', 'w') as f:
                arr = [
                    '"tr_group_input",' + ','.join(str(v) for v in model.tr_group_input.detach().cpu().numpy().tolist()),
                    '"tr_group_inter",' + ','.join(str(v) for v in model.tr_group_inter.detach().cpu().numpy().tolist()),
                    '"tr_group_output",' + ','.join(str(v) for v in model.tr_group_output.detach().cpu().numpy().tolist()),
                    '"tr_head_inter1",' + ','.join(str(v) for v in model.tr_head_inter.detach().cpu().numpy().tolist()),
                    '"tr_head_inter2",' + ','.join(str(v) for v in ((tr_head_inter[0] << tr_head_inter[1]).astype(np.float64) / head_inter_linear.factors).tolist()),
                    '"tr_head_output1",' + ','.join(str(v) for v in model.tr_head_output.detach().cpu().numpy().tolist()),
                    '"tr_head_output2",' + ','.join(str(v) for v in x.tolist()),
                ]
                f.write('\n'.join(arr))

            prob_x = softmax(x_fixed)
            #prob_x = F.softmax(torch.from_numpy(x / heat), dim=0).detach().cpu().numpy()
            prob_yy = F.softmax(torch.from_numpy(yy / heat), dim=0).detach().cpu().numpy()
            prob_x = prob_x / np.average(prob_x) * np.average(prob_yy)
            #pprint((yy))
            #pprint((x, yy, np.round((x - yy) / yy * 100).astype(np.int32)))
            #pprint((prob_x, prob_yy, np.round((prob_x - prob_yy) / prob_yy * 100).astype(np.int32)))
            w = 0
            x = (w * x + (1 - w) * yy)

        # x /= heat
        # output_tensor = F.softmax(torch.from_numpy(x), dim=0)
        # Generate a random letter based on the probabilities
        output_tensor = torch.from_numpy(softmax(x_fixed).astype(np.float32))
        letter_index = torch.multinomial(output_tensor, 1).item()
        return lang.INDEX_TO_LETTER[letter_index]

    def generate(count: int, use_float: bool):
        context = 'lorem ipsum '
        text = ''
        for i in range(count):
            next_letter = next_char(context, use_float)
            text += next_letter
            context = context[1:] + next_letter
        return text

    print('--------------------------')
    #print(next_char('lorem ipfum '))
    #print(next_char('nerido tet e', True))
    #print(next_char('late et max ', True))
    print(generate(300, False))
    print(generate(300, True))


if __name__ == '__main__':
    test()
