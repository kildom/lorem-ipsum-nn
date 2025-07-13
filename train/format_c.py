
import re
from pathlib import Path

from lang import get_languages

def c_str(s: str) -> str:
    literal = re.match(r'^[ a-z]+$', s)
    if literal:
        return s
    else:
        b = s.encode('utf-8')
        return ''.join(f'\\x{c:02x}' for c in b)

def format_matrix(value: list[list[int]], second_dim=False) -> str:
    if second_dim:
        return "{\n    { " + " },\n    { ".join(', '.join((str(m) for m in n)) for n in value) + " },\n}"
    else:
        return "{\n    " + ",\n    ".join(', '.join((str(m) for m in n)) for n in value) + ",\n}"

def format_vector(value: list[int]) -> str:
    return "{\n    " + ", ".join(str(n) for n in value) + ",\n}"

def format_layer(layer: dict, name: str) -> str:
    global_scope = ''
    local_scope = ''
    if layer['type'] == 'relu':
        local_scope += f'static const struct LoremIpsumReLU {name} = {{\n'
        local_scope += '    .type = LOREM_IPSUM_LAYER_RELU,\n'
        local_scope += '};\n\n'
    elif layer['type'] == 'bit_shift':
        global_scope += f'static const uint8_t {name}_value[] = {format_vector(layer["value"])};\n\n'
        local_scope += f'static const struct LoremIpsumBitShift {name} = {{\n'
        local_scope += '    .type = LOREM_IPSUM_LAYER_BIT_SHIFT,\n'
        local_scope += f'    .value = {name}_value,\n'
        local_scope += '};\n\n'
    elif layer['type'] == 'linear':
        global_scope += f'static const int8_t {name}_weight[] = {format_matrix(layer["weight"])};\n\n'
        global_scope += f'static const int32_t {name}_bias[] = {format_vector(layer["bias"])};\n\n'
        local_scope += f'static const struct LoremIpsumLinear {name} = {{\n'
        local_scope += '    .type = LOREM_IPSUM_LAYER_LINEAR,\n'
        local_scope += f'    .weight = {name}_weight,\n'
        local_scope += f'    .bias = {name}_bias,\n'
        local_scope += f'    .output_size = {len(layer["weight"])},\n'
        local_scope += '};\n\n'
    global_scope += local_scope
    return global_scope

def format_nn(nn: list[dict], name: str) -> str:
    global_scope = ''
    local_scope = ''
    local_scope += f'static const void* const {name} = {{\n'
    for i, layer in enumerate(nn):
        layer_name = f"{name}_{i}_{layer['type']}"
        global_scope += format_layer(layer, layer_name)
        local_scope += f'    (const void*)&{layer_name},\n'
    local_scope += f'}};\n\n'
    global_scope += local_scope
    return global_scope

def format_c(output_model: dict, file: Path):
    file = file.with_suffix('.c')
    global_scope = '#include "lorem-ipsum-int.h"\n\n'

    local_scope = f'const struct LoremIpsumModel lorem_ipsum_{output_model["lang"]} = {{\n'

    local_scope += f'    .lang = "{c_str(output_model["lang"])}",\n'

    global_scope += f'static const char* const model_letters[] = {{\n    '
    global_scope += ', '.join(f'"{c_str(letter)}"' for letter in output_model['letters'])
    global_scope += f',\n}};\n\n'
    local_scope += f'    .letters = model_letters,\n'

    local_scope += f'    .letters_count = {len(output_model["letters"])},\n'

    global_scope += f'static const uint8_t model_letters_embedding[] = {format_matrix(output_model["letters_embedding"])};\n\n'
    local_scope += f'    .letters_embedding = model_letters_embedding,\n'
    local_scope += f'    .letter_embedding_length = {len(output_model["letters_embedding"][0])},\n'

    global_scope += format_nn(output_model['group'], "model_group")
    local_scope += f'    .group = model_group,\n'

    global_scope += format_nn(output_model['head'], "model_head")
    local_scope += f'    .head = model_head,\n'

    global_scope += f'static const int32_t model_output_scale_u0_31[] = {format_vector(output_model["output_scale_u0_31"])};\n\n'
    local_scope += f'    .output_scale_u0_31 = model_output_scale_u0_31,\n'
    local_scope += f'    .output_shift = {output_model["output_shift"]},\n'
    local_scope += f'    .fractional_bits = {output_model["fractional_bits"]},\n'

    local_scope += f'    .prob_fractional_bits = {output_model["prob_fractional_bits"]},\n'
    global_scope += f'static const int32_t model_prob_exp_table[][8] = {format_matrix(output_model["prob_exp_table"], True)};\n\n'
    local_scope += f'    .prob_exp_table = model_prob_exp_table,\n'
    local_scope += f'    .prob_exp_table_rows = {len(output_model["prob_exp_table"])},\n'

    global_scope += f'static const int32_t model_empty_group_embedding[] = {format_vector(output_model["empty_group_embedding"])};\n\n'
    local_scope += f'    .empty_group_embedding = model_empty_group_embedding,\n'

    global_scope += f'static const uint16_t model_prob_dot[] = {format_vector(output_model["prob_dot"])};\n\n'
    local_scope += f'    .prob_dot = model_prob_dot,\n'
    local_scope += f'    .prob_dot_length = {len(output_model["prob_dot"])},\n'

    prob_comma = output_model["prob_comma"]
    max_len = max(len(row) for row in prob_comma)
    prob_comma = [row + [0] * (max_len - len(row)) for row in prob_comma]
    global_scope += f'static const uint16_t model_prob_comma[] = {format_matrix(prob_comma)};\n\n'
    local_scope += f'    .prob_comma = model_prob_comma,\n'
    local_scope += f'    .prob_comma_length = {max_len},\n'

    local_scope += '};\n\n'
    global_scope += local_scope
    file.write_text(global_scope)

    header_scope = '#include "lorem-ipsum-int.h"\n\n'
    header_scope += f'#define LOREM_IPSUM_MODELS {", ".join(("&lorem_ipsum_" + lang) for lang in get_languages())}\n\n'
    for lang in get_languages():
        header_scope += f'extern const struct LoremIpsumModel lorem_ipsum_{lang};\n'
    header_file = file.parent / 'lorem-ipsum-models.h'
    header_file.write_text(header_scope)
