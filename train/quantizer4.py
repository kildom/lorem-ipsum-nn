
import json
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import random
from pathlib import Path
from lang import get_lang_config
from model import GeneratorSharedNet
from pprint import pprint
from train import train
import torch.nn.functional as F

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
                 input_shift: 'npt.NDArray[np.int32]',
                 max_input: float,
                 factors: 'npt.NDArray[np.float64]'=None,
                 ):
        self.weight = weight
        self.bias = bias
        self.input_shift = input_shift
        self.max_input = max_input
        self.factors = factors

    def forward(self, x: 'npt.NDArray[np.int32]', bit_shift: int) -> 'tuple[npt.NDArray[np.int32], int]':
        assert np.max(np.abs(x)) < 2147483648
        x = x >> self.input_shift
        actual_max = np.max(np.abs(x))
        current_shift = 0
        while (actual_max >> current_shift) > self.max_input:
            current_shift += 1
        total_shift = bit_shift + current_shift
        y = (self.bias >> total_shift) + np.dot(self.weight.astype(np.int32), x.astype(np.int32) >> current_shift)
        assert np.max(np.abs(self.bias)) < 1073741824
        assert np.max(np.abs(np.dot(self.weight, x >> current_shift))) < 1073741824
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

def estimate_input(values: 'npt.NDArray[np.float64]', shifts: 'npt.NDArray[np.int32]'):
    max_value = np.max(np.abs(values) / (1 << shifts).reshape(-1, 1), axis=0)
    half_range = max_value / 2
    v = np.maximum(np.abs(values), half_range)
    v = np.average(v, axis=0)
    return v

def quantize_linear(linear: nn.Linear, input_factors: 'npt.NDArray[np.float64]', estimated_input: 'npt.NDArray[np.float64]') -> QuantizedLinear:
    initial_weight = linear.weight.detach().cpu().numpy().astype(np.float64)
    initial_bias = linear.bias.detach().cpu().numpy().astype(np.float64)
    input_shift = np.zeros(len(input_factors), dtype=np.int32)
    while True:
        # Get weight
        weight = initial_weight * (1 << input_shift) / input_factors
        # Maximize at least one weight to 127 in each row by calculating output factors
        max_weight = np.max(np.abs(weight), axis=1)
        max_max_weight  = np.max(max_weight)
        max_weight = np.maximum(max_weight, max_max_weight / 1000000)
        output_factors = 127 / max_weight
        weight = weight * output_factors.reshape(-1, 1)
        # If possible, maximize weights in each column by shifting right the input
        max_weight = np.max(np.abs(weight), axis=0)
        can_be_updated = ((estimated_input / (1 << input_shift) / max_weight > 6) & (max_weight <= 63)).astype(np.int32)
        if max(can_be_updated) > 0:
            input_shift += can_be_updated
            continue
        # Calculate final quantized weight
        weight = np.round(weight)
        assert np.max(np.abs(weight)) <= 127
        weight = weight.astype(np.int32)
        # Calculate final quantized bias
        bias = initial_bias * output_factors
        bias = np.round(bias).astype(np.int32)
        # Calculate maximum input value that this layer can handle without overflows
        negatives = np.minimum(0, weight)
        negatives_max = np.max(np.abs(np.sum(negatives, axis=1)))
        #negatives_max = int(negatives_max)
        positives = np.maximum(0, weight)
        positives_max = np.max(np.abs(np.sum(positives, axis=1)))
        #positives_max = int(positives_max)
        max_input = 1073741823 // int(max(negatives_max, positives_max))
        # If bias get too large, we need to increase input_shift and try again or reduce output_factors[i]
        # The problem is which of those two to choose. For now, the models are not hitting this limit,
        # so it is not implemented.
        assert np.max(np.abs(bias)) <= 1073741823, f'Bias {np.max(np.abs(bias))} is too large, bit-shifting not implemented.'
        bias = bias.astype(np.int64) # Why not int32?
        pprint((output_factors, weight, bias, initial_bias, max_weight, input_shift, max_input))
        break
    #print(weight, bias, input_size, output_size)
    return QuantizedLinear(weight, bias, input_shift, max_input, output_factors)


def test():
    lang_code = 'la'
    lang = get_lang_config(lang_code)
    MODEL_PATH = Path(__file__).parent.parent / f'data/{lang_code}/basic-model.pt'
    model = GeneratorSharedNet(lang, GeneratorSharedNet.ALL, False).eval()
    model.load_state_dict(torch.load(MODEL_PATH))

    letter_embedding_precise = model.get_letter_embedding().detach().cpu().numpy()
    max_letter_embedding = np.max(letter_embedding_precise, axis=0)
    letter_factors = 255 / max_letter_embedding
    letter_embedding = np.round(letter_embedding_precise * letter_factors) / letter_factors
    pprint((letter_embedding, letter_embedding * letter_factors, estimate_input(letter_embedding * letter_factors, np.zeros(len(letter_embedding), dtype=np.int32))))

    letters_per_group = model.group_inter_linear.in_features // len(letter_embedding[0])
    groups_per_context = model.head_inter_linear.in_features // model.group_output_linear.out_features
    relu = QuantizedReLU()

    # Quantize group intermediate layer
    group_inter_linear = quantize_linear(
        model.group_inter_linear,
        np.concat([letter_factors] * letters_per_group),
        np.concat([estimate_input(letter_embedding * letter_factors, np.zeros(len(letter_embedding), dtype=np.int32))] * letters_per_group)
        )
    layer1_outputs = np.zeros((10000, model.group_inter_linear.out_features), dtype=np.int32)
    layer1_shifts = np.zeros(len(layer1_outputs), dtype=np.int32)
    for i in range(layer1_outputs.shape[0]):
        x = np.concatenate([letter_embedding[int(x) % lang.ALPHABET_LENGTH] * letter_factors for x in random.randbytes(letters_per_group)])
        x = np.round(x).astype(np.int32)
        layer1_outputs[i], layer1_shifts[i] = relu(*group_inter_linear(x, 0))
    pprint(('group_inter_linear estimated output', estimate_input(layer1_outputs, layer1_shifts)))

    # Quantize group output layer
    group_output_linear = quantize_linear(
        model.group_output_linear,
        group_inter_linear.factors,
        estimate_input(layer1_outputs, layer1_shifts)
        )
    layer2_outputs = np.zeros((len(layer1_outputs), model.group_output_linear.out_features), dtype=np.int32)
    layer2_shifts = np.zeros(len(layer1_outputs), dtype=np.int32)
    for i in range(layer1_outputs.shape[0]):
        layer2_outputs[i], layer2_shifts[i] = relu(*group_output_linear(layer1_outputs[i], layer1_shifts[i]))
    pprint(('group_output_linear estimated output', estimate_input(layer2_outputs, layer2_shifts)))

    # Quantize head intermediate layer
    head_inter_linear = quantize_linear(
        model.head_inter_linear,
        np.concatenate([group_output_linear.factors] * groups_per_context),
        np.concatenate([estimate_input(layer2_outputs, layer2_shifts)] * groups_per_context)
        )
    layer3_outputs = np.zeros((len(layer1_outputs) // groups_per_context, model.head_inter_linear.out_features), dtype=np.int32)
    layer3_shifts = np.zeros(len(layer1_outputs) // groups_per_context, dtype=np.int32)
    for i in range(layer3_outputs.shape[0]):
        input_values = np.concatenate(layer2_outputs[i * groups_per_context:(i + 1) * groups_per_context])
        input_shifts = layer2_shifts[i * groups_per_context:(i + 1) * groups_per_context]
        assert np.min(input_shifts) == np.max(input_shifts)
        layer3_outputs[i], layer3_shifts[i] = relu(*head_inter_linear(input_values, input_shifts[0]))
    pprint(('group_output_linear estimated output', estimate_input(layer3_outputs, layer3_shifts)))


    # Quantize head output layer
    head_output_linear = quantize_linear(
        model.head_output_linear,
        head_inter_linear.factors,
        estimate_input(layer3_outputs, layer3_shifts)
        )
    layer4_outputs = np.zeros((len(layer3_outputs), model.head_output_linear.out_features), dtype=np.int32)
    layer4_shifts = np.zeros(len(layer3_outputs), dtype=np.int32)
    for i in range(layer3_outputs.shape[0]):
        layer4_outputs[i], layer4_shifts[i] = relu(*head_output_linear(layer3_outputs[i], layer3_shifts[i]))
    pprint(('group_output_linear estimated output', estimate_input(layer4_outputs, layer4_shifts)))

    def next_char(text: str, use_float) -> 'tuple[str, int]':
        input1 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[0]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[1]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[2]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[3]]] * letter_factors).astype(np.int32),
        ])
        input2 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[4]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[5]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[6]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[7]]] * letter_factors).astype(np.int32),
        ])
        input3 = np.concatenate([
            (letter_embedding[lang.LETTER_TO_INDEX[text[8]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[9]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[10]]] * letter_factors).astype(np.int32),
            (letter_embedding[lang.LETTER_TO_INDEX[text[11]]] * letter_factors).astype(np.int32),
        ])

        def group(input: 'npt.NDArray[np.int32]') -> 'tuple[npt.NDArray[np.int32], int]':
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
        tr_head_inter = x, sh
        x, sh = head_output_linear(x, sh)

        #assert sh == 0, f'Head shift {sh} is not zero.'

        x = (x << sh).astype(np.float64) / head_output_linear.factors
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
            pprint((x, yy, np.round((x - yy) / yy * 100).astype(np.int32), sh))
            w = 0
            x = (w * x + (1 - w) * yy)

        heat = 0.6
        x /= heat
        output_tensor = F.softmax(torch.from_numpy(x), dim=0)
        # Generate a random letter based on the probabilities
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
    print(next_char('e mentislo p', True))
    #print(next_char('late et max '))
    #print(generate(250, False))
    #print(generate(250, True))


if __name__ == '__main__':
    test()
