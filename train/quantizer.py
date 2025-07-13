
import torch.nn as nn
import numpy as np
import numpy.typing as npt


def linear_min_max_weight(matrix: 'npt.NDArray[np.float32]', bias: 'npt.NDArray[np.float32]') -> 'tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]':
    output_size = matrix.shape[0]
    min_value = np.full(output_size, np.inf, dtype=np.float32)
    max_value = np.full(output_size, -np.inf, dtype=np.float32)
    for row in range(output_size):
        min_value[row] = min(min_value[row], bias[row], np.min(matrix[row, :]))
        max_value[row] = max(max_value[row], bias[row], np.max(matrix[row, :]))
    return min_value, max_value


def scale_int8(value, factor: float) -> float:
    value = float(value) * factor
    value = min(max(round(value), -128), 127)
    return value


def scale_int24(value, factor: float) -> float:
    value = float(value) * factor
    value = min(max(round(value), -16777216), 16777215)
    return value


def quantize_linear_to_int8(linear_layer: nn.Linear, input_range: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.float32]':

    '''
    Calculates quantize version of a linear layer to fit into int8 for weight and int32 for bias.

    input_range is a 2D array of shape (input_size, 2) with the minimum and maximum values (inclusive) for each input feature.
    Or two-element array with the same value for all features, e.g. [0, 255].

    returns weight, bias, input_shifts, factors, output_range
    * weight - quantized weight matrix (int8)
    * bias - quantized bias vector (int32)
    * input_shifts - how much to bit-shift input values before multiplying by the weight
    * factors - scaling factors for each output value caused by the quantization.
      The output must be divided by these factors to get the original values.
    * output_range - a 2D array of shape (output_size, 2) with the minimum and maximum values (inclusive) for each output feature.
    '''

    input_size = linear_layer.in_features
    output_size = linear_layer.out_features
    input_shifts = [0] * input_size

    if input_range.shape == (2,):
        input_range = np.column_stack((
            np.full(input_size, input_range[0], dtype=np.int32),
            np.full(input_size, input_range[1], dtype=np.int32)
            ))

    while True:
        # Get the data from the linear layer.
        # Indexing the matrix: weight[output_size][input_size] == weight[row][column] ==  weight[y][x]
        weight = linear_layer.weight.detach().cpu().numpy().copy()
        bias = linear_layer.bias.detach().cpu().numpy().copy()

        # Adjust weight to take into account the input bit shifts.
        for row in range(output_size):
            for column in range(input_size):
                weight[row][column] *= 1 << input_shifts[column]

        # Get values range and calculate the scaling factors allowing a full range of int8 (and int32 for bias).
        min_value, max_value = linear_min_max_weight(weight, bias / 65536) # bias will be 24-bit, so factors can be bigger for it
        min_factor = 128 / np.maximum(1e-20, np.abs(min_value))
        max_factor = 127 / np.maximum(1e-20, np.abs(max_value))
        factors: 'npt.NDArray[np.float32]' = np.minimum(min_factor, max_factor)
        # pprint((min_value, max_value, factors, min_value * factors, max_value * factors))
        # pprint((linear_layer.weight, linear_layer.bias))

        # Scale the weights and bias to fit into int8 and int32.
        for row in range(output_size):
            factor = float(factors[row])
            for column in range(input_size):
                weight[row][column] = scale_int8(weight[row][column], factor)
            bias[row] = scale_int24(bias[row], factor)
        weight = weight.astype(np.int32)
        bias = bias.astype(np.int32)
        # pprint((weight, bias))

        # Calculate maximum integer values when calculating the layer output. It must fit into int32.
        # If they are too big, request smaller input ranges.
        reduce_input = False
        min_terms = np.full((output_size, input_size), 0, dtype=np.int64)
        max_terms = np.full((output_size, input_size), 0, dtype=np.int64)
        output_range = np.full((output_size, 2), 0, dtype=np.int64)
        for row in range(output_size):
            for column in range(input_size):
                if weight[row][column] < 0:
                    min_terms[row][column] = int(weight[row][column]) * int(input_range[column][1] >> input_shifts[column])
                    max_terms[row][column] = int(weight[row][column]) * int(input_range[column][0] >> input_shifts[column])
                else:
                    min_terms[row][column] = int(weight[row][column]) * int(input_range[column][0] >> input_shifts[column])
                    max_terms[row][column] = int(weight[row][column]) * int(input_range[column][1] >> input_shifts[column])
            min_value = int(min_terms[row].sum()) + int(bias[row])
            max_value = int(max_terms[row].sum()) + int(bias[row])
            reduce_input = reduce_input or (min_value < -2147483648) or (max_value > 2147483647)
            output_range[row][0] = min_value
            output_range[row][1] = max_value
        if not reduce_input:
            break
        max_index = np.unravel_index(np.argmax(max_terms), max_terms.shape)
        max_term_value = max_terms[max_index]
        min_index = np.unravel_index(np.argmin(min_terms), min_terms.shape)
        min_term_value = min_terms[min_index]
        if abs(max_term_value) > abs(min_term_value):
            index = max_index
        else:
            index = min_index
        min_shift = min(input_shifts)
        new_shift = input_shifts[index[1]] + 1
        if new_shift > min_shift + 1:
            for i in range(input_size):
                input_shifts[i] = min_shift + 1
        else:
            input_shifts[index[1]] = new_shift

    return weight, bias, np.array(input_shifts, dtype=np.int32), factors, output_range.astype(np.int32)


class SimpleQuantizedModel:
    def __call__(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        return self.forward(x)

    def store(self) -> 'list[dict]':
        raise NotImplementedError("Subclasses must implement the store method to save the model state.")


class QuantizedLinear(SimpleQuantizedModel):

    def __init__(self,
                 weight_or_linear_layer: 'npt.NDArray[np.int32]',
                 bias_or_input_range: 'npt.NDArray[np.int32]',
                 input_shifts: 'npt.NDArray[np.int32]'=None,
                 factors: 'npt.NDArray[np.int32]'=None,
                 output_range: 'npt.NDArray[np.int32]'=None
                 ):
        if input_shifts is None:
            weight, bias, input_shifts, factors, output_range = quantize_linear_to_int8(weight_or_linear_layer, bias_or_input_range)
        else:
            weight = weight_or_linear_layer
            bias = bias_or_input_range
        self.weight = weight
        self.bias = bias
        self.input_shifts = input_shifts
        self.factors = factors
        self.output_range = output_range

    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        x = x >> self.input_shifts
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


class QuantizedReLU(SimpleQuantizedModel):

    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        return np.maximum(0, x)

    def store(self) -> 'list[dict]':
        return [{
            'type': 'relu'
        }]


class QuantizedSequential(SimpleQuantizedModel):

    def __init__(self, *models):
        self.models = models
        
    def forward(self, x: 'npt.NDArray[np.int32]') -> 'npt.NDArray[np.int32]':
        for model in self.models:
            x = model(x)
        return x
