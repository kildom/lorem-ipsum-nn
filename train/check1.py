
import json
import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from pathlib import Path
from model import GROUPS_PER_CONTEXT, LETTERS_PER_GROUP, GeneratorSharedNet, LETTERS_PER_CONTEXT, LETTER_EMBEDDING_SIZE
from lang import get_lang_config
from quantizer import QuantizedLinear, QuantizedReLU, QuantizedSequential, SimpleQuantizedModel

lang_code = 'la'
lang = get_lang_config(lang_code)

conf_names = [
    'ALL',
    'NO_LETTER',
    'NO_GROUP_INTER',
    'NO_GROUP',
    'NO_HEAD_INTER',
]

def check_model(layers_conf):

    if layers_conf == GeneratorSharedNet.NO_LETTER:
        MODEL_PATH = Path(__file__).parent.parent / f'data/{lang_code}/direct-model.pt'
    else:
        MODEL_PATH = Path(__file__).parent.parent / f'data/{lang_code}/basic-model.pt'
    
    json_model = json.loads((Path(__file__).parent.parent / f'models/{lang_code}.json').read_text())
    letters_embedding = np.array(json_model['letters_embedding'], dtype=np.float32)

    model = GeneratorSharedNet(lang, layers_conf, False)
    model.load_state_dict(torch.load(MODEL_PATH))

    def text_to_torch(text: str) -> torch.Tensor:
        if layers_conf == GeneratorSharedNet.ALL:
            result = np.zeros(lang.ALPHABET_LENGTH * LETTERS_PER_CONTEXT, dtype=np.float32)
            for i, letter in enumerate(text.lower()):
                index = lang.LETTER_TO_INDEX[letter]
                result[i * lang.ALPHABET_LENGTH + index] = 1.0
        else:
            result = np.zeros(LETTERS_PER_CONTEXT * LETTER_EMBEDDING_SIZE, dtype=np.float32)
            for i, letter in enumerate(text.lower()):
                index = lang.LETTER_TO_INDEX[letter]
                result[i * LETTER_EMBEDDING_SIZE:(i + 1) * LETTER_EMBEDDING_SIZE] = letters_embedding[index]
        return torch.from_numpy(result)

    def generate_next(text: str) -> str:
        input_tensor = text_to_torch(text)
        output_tensor = model(input_tensor)
        heat = 0.6
        output_tensor /= heat
        output_tensor = F.softmax(output_tensor, dim=0)
        # Generate a random letter based on the probabilities
        letter_index = torch.multinomial(output_tensor, 1).item()
        return lang.INDEX_TO_LETTER[letter_index]

    model.eval()
    with torch.no_grad():
        text = 'lorem ipsum '
        output = ''
        for _ in range(100):
            x = generate_next(text)
            output += x
            text = text[1:] + x
        print(f'{conf_names[layers_conf]:14}: {output}')

check_model(GeneratorSharedNet.ALL)
check_model(GeneratorSharedNet.NO_LETTER)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def check_quantized():

    json_model = json.loads((Path(__file__).parent.parent / f'models/{lang_code}.json').read_text())
    letters_embedding = np.array(json_model['letters_embedding'], dtype=np.int32)
    group = QuantizedSequential.load(json_model['group'])
    head = QuantizedSequential.load(json_model['head'])
    output_scale_u0_31 = np.array(json_model['output_scale_u0_31'], dtype=np.int64)

    def text_to_torch(text: str) -> 'npt.NDArray[np.int32]':
        result = np.zeros(LETTERS_PER_CONTEXT * LETTER_EMBEDDING_SIZE, dtype=np.int32)
        for i, letter in enumerate(text.lower()):
            index = lang.LETTER_TO_INDEX[letter]
            result[i * LETTER_EMBEDDING_SIZE:(i + 1) * LETTER_EMBEDDING_SIZE] = letters_embedding[index]
        return result

    def generate_next(text: str) -> str:
        input_tensor = text_to_torch(text)
        groups_list = []
        for i in range(GROUPS_PER_CONTEXT):
            groups_list.append(group(input_tensor[i * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE:(i + 1) * LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE]))
        output_tensor = head(np.concatenate(groups_list))
        output_tensor = (output_tensor.astype(np.int64) * output_scale_u0_31) >> json_model['output_shift']
        output_tensor = output_tensor.astype(np.float32)
        heat = 0.6
        output_tensor /= heat
        output_tensor = softmax(output_tensor)
        # Generate a random letter based on the probabilities
        letter_index = torch.multinomial(torch.from_numpy(output_tensor), 1).item()
        return lang.INDEX_TO_LETTER[letter_index]

    text = 'lorem ipsum '
    output = ''
    for _ in range(100):
        x = generate_next(text)
        output += x
        text = text[1:] + x
    print(f'QUANTIZED     : {output}')

check_quantized()
