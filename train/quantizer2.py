
import json
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from pathlib import Path
from lang import get_lang_config
from model import GeneratorSharedNet
from pprint import pprint

np.set_printoptions(
    linewidth=150,  # wider output
    threshold=np.inf,  # print entire array, no summarizing
    precision=4,  # 4 decimal places
    suppress=True  # don't use scientific notation for small values
)


class SimpleQuantizedModel:
    def __call__(self, x: 'npt.NDArray[np.int32]', bit_shift: int) -> 'tuple[npt.NDArray[np.int32], int]':
        return self.forward(x, bit_shift)

    def store(self) -> 'list[dict]':
        raise NotImplementedError("Subclasses must implement the store method to save the model state.")


class QuantizedLinear(SimpleQuantizedModel):

    def __init__(self,
                 weight: 'npt.NDArray[np.int32]',
                 bias: 'npt.NDArray[np.int32]',
                 max_input: float,
                 factors: 'npt.NDArray[np.float64]'=None,
                 ):
        self.weight = weight
        self.bias = bias
        self.max_input = max_input
        self.factors = factors

    def forward(self, x: 'npt.NDArray[np.int32]', bit_shift: int) -> 'tuple[npt.NDArray[np.int32], int]':
        actual_max = np.max(np.abs(x))
        current_shift = 0
        while (actual_max >> current_shift) > self.max_input:
            current_shift += 1
        total_shift = bit_shift + current_shift
        y = (self.bias >> total_shift) + np.dot(self.weight, x >> current_shift)
        return (y, total_shift)

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

    def forward(self, x: 'npt.NDArray[np.int32]', bit_shift: int) -> 'tuple[npt.NDArray[np.int32], int]':
        return (np.maximum(0, x), bit_shift)

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


def quantize_linear(linear: nn.Linear, input_factors: 'npt.NDArray[np.float64]') -> QuantizedLinear:
    initial_weight = linear.weight.detach().cpu().numpy().astype(np.float64)
    initial_bias = linear.bias.detach().cpu().numpy().astype(np.float64)
    input_size = linear.in_features
    output_size = linear.out_features
    input_shift = 0
    while True:
        weight = initial_weight * (1 << input_shift) / input_factors
        max_weight = np.max(np.abs(weight), axis=1)
        max_max_weight  = np.max(max_weight)
        max_weight = np.maximum(max_weight, max_max_weight / 1000000)
        output_factors = 127 / max_weight
        weight = weight * output_factors.reshape(-1, 1)
        weight = np.round(weight).astype(np.int64)
        bias = initial_bias * output_factors
        bias = np.round(bias).astype(np.int64)
        negatives = np.minimum(0, weight)
        negatives_max = int(np.max(np.abs(np.sum(negatives, axis=1))))
        positives = np.maximum(0, weight)
        positives_max = int(np.max(np.abs(np.sum(positives, axis=1))))
        max_input = 1073741823 // max(negatives_max, positives_max)
        pprint((weight, bias, max_input, output_factors))
        # If bias get too large, we need to increase input_shift and try again or reduce output_factors[i]
        # The problem is which of those two to choose. For now, the models are not hitting this limit,
        # so it is not implemented.
        assert np.max(np.abs(bias)) <= 1073741823, f'Bias {np.max(np.abs(bias))} is too large, bit-shifting not implemented.'
        break
    #print(weight, bias, input_size, output_size)
    return QuantizedLinear(weight, bias, max_input, output_factors)



lang_code = 'la'
lang = get_lang_config(lang_code)

MODEL_PATH = Path(__file__).parent.parent / f'data/{lang_code}/direct-model.pt'

model = GeneratorSharedNet(lang, GeneratorSharedNet.NO_LETTER, False)
model.load_state_dict(torch.load(MODEL_PATH))
json_model = json.loads((Path(__file__).parent.parent / f'models/{lang_code}.json').read_text())
letters_embedding = np.array(json_model['letters_embedding'], dtype=np.int64)

relu = QuantizedReLU()
group_inter_linear = quantize_linear(model.group_inter_linear, np.ones(model.group_inter_linear.in_features, dtype=np.float64))
group_output_linear = quantize_linear(model.group_output_linear, group_inter_linear.factors)
head_inter_linear = quantize_linear(model.head_inter_linear, np.concat([group_output_linear.factors] * (model.head_inter_linear.in_features // model.group_output_linear.out_features)))
head_output_linear = quantize_linear(model.head_output_linear, head_inter_linear.factors)

fraction_bits = 0
while (1 << (fraction_bits + 1)) / np.min(head_output_linear.factors) <= 1073741823:
    fraction_bits += 1

factors_fixed = np.round((1 << fraction_bits) / head_output_linear.factors).astype(np.int64)

pprint((head_output_linear.factors, factors_fixed, fraction_bits))

def next_char(text: str) -> 'tuple[str, int]':
    input1 = np.concatenate([
        letters_embedding[lang.LETTER_TO_INDEX[text[0]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[1]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[2]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[3]]],
    ])
    input2 = np.concatenate([
        letters_embedding[lang.LETTER_TO_INDEX[text[4]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[5]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[6]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[7]]],
    ])
    input3 = np.concatenate([
        letters_embedding[lang.LETTER_TO_INDEX[text[8]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[9]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[10]]],
        letters_embedding[lang.LETTER_TO_INDEX[text[11]]],
    ])

    def group(input: 'npt.NDArray[np.int64]') -> 'tuple[npt.NDArray[np.int64], int]':
        x = input
        sh = 0
        x, sh = group_inter_linear(x, sh)
        x, sh = relu(x, sh)
        x, sh = group_output_linear(x, sh)
        x, sh = relu(x, sh)
        return x, sh

    x1, sh1 = group(input1)
    x2, sh2 = group(input2)
    x3, sh3 = group(input3)

    assert max(sh1, sh2, sh3) == 0, f'Group shift {max(sh1, sh2, sh3)} is not zero.'

    x = np.concatenate([x1, x2, x3])
    x, sh = head_inter_linear(x, sh1)
    x, sh = relu(x, sh)
    x, sh = head_output_linear(x, sh)

    #assert sh == 0, f'Head shift {sh} is not zero.'

    x = x * factors_fixed / (1 << (fraction_bits - sh))
    with torch.no_grad():
        xx = torch.from_numpy(np.concatenate([input1, input2, input3]).astype(np.float32))
        model.eval()
        yy = model(xx)
        yy = yy.detach().cpu().numpy()

    pprint((x, yy, np.round((x - yy) / yy * 100).astype(np.int64), sh))


print('--------------------------')
#print(next_char('lorem ipfum '))
#print(next_char('some at rand'))
print(next_char('lat en et ma'))
