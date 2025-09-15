
import re
from pathlib import Path
from textwrap import dedent
from lang import get_lang_config, get_languages

def c_str(s: str) -> str:
    literal = re.match(r'^[ a-zA-Z]+$', s)
    if literal:
        return s
    else:
        b = s.encode('utf-8')
        return ''.join(f'\\x{c:02x}' for c in b)

def format_matrix(value: list[list[int]], second_dim=False) -> str:
    if second_dim:
        return "{\n            { " + " },\n            { ".join(', '.join((str(int(m)) for m in n)) for n in value) + " },\n        }"
    else:
        return "{\n            " + ",\n            ".join(', '.join((str(int(m)) for m in n)) for n in value) + ",\n        }"

def transpose(value: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*value)]

def format_vector(value: list[int]) -> str:
    return "{\n            " + ", ".join(str(int(n)) for n in value) + ",\n        }"

def format_layer(layer: dict, id: str, name: str) -> str:
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

nl = '\n'
sl = '\\'


def format_linear(layer: dict, id: str, name: str) -> str:
    return f'''
        static const int8_t {id}{name}_weight[] = {format_matrix(layer["weight"])};

        static const int32_t {id}{name}_bias[] = {format_vector(layer["bias"])};

        static const uint8_t {id}{name}_input_shift[] = {format_vector(layer["input_shift"])};

        static const int32_t {id}{name}_input_clamp[][2] = {format_matrix(transpose(layer["input_clamp"]), 2)};

        static const struct LoremIpsumLinear {id}{name} = {{
            .type = LOREM_IPSUM_LAYER_LINEAR,
            .weight = {id}{name}_weight,
            .bias = {id}{name}_bias,
            .input_shift = {id}{name}_input_shift,
            .input_clamp = {id}{name}_input_clamp,
            .input_size = {len(layer["weight"][0])},
            .output_size = {len(layer["weight"])},
        }};
        '''

def format_relu(layer: dict, id: str, name: str) -> str:
    return f'''
        static const struct LoremIpsumReLU {id}{name} = {{
            .type = LOREM_IPSUM_LAYER_RELU,
            .input_size = {layer['input_size']},
        }};
        '''

def format_scaled_softmax(layer: dict, id: str, name: str) -> str:
    return f'''
        static const int32_t {id}{name}_weight[] = {format_vector(layer["weight"])};

        static const struct LoremIpsumScaledSoftmax {id}{name} = {{
            .type = LOREM_IPSUM_LAYER_SCALED_SOFTMAX,
            .weight = {id}{name}_weight,
            .frac_bits = {layer["frac_bits"]},
            .input_size = {len(layer['weight'])},
        }};
        '''


def format_nn(nn: list[dict], id: str, name: str) -> str:
    return f'''
        {''.join(globals()[f'format_{layer["type"]}'](layer, id, f'{name}_{i}_{layer["type"]}') for i, layer in enumerate(nn))}

        static const void* const {id}{name}[] = {{
            {f',{nl}            '.join(f'&{id}{name}_{i}_{layer["type"]}' for i, layer in enumerate(nn))},
            (void*)0,
        }};
        '''


def format_c(output_model: dict, file: Path):

    file = file.with_suffix('.c')
    file = Path(__file__).parent.parent / ('src/c_lib/src/' + file.with_suffix('.c').name)
    id = 'model_' + output_model["lang"] + '_'

    text = f'''
        #include <stdint.h>
        #include "lorem-ipsum-int.h"

        static const char* const {id}lower_letters[] = {{
            {', '.join(f'"{c_str(letter)}"' for letter in output_model['letters'])},
        }};

        static const char* const {id}upper_letters[] = {{
            {', '.join(f'"{c_str(letter)}"' for letter in output_model['letters'].upper())},
        }};

        static const uint8_t {id}letters_embedding[] = {format_matrix(output_model["letters_embedding"])};

        {format_nn(output_model['group'], id, 'group')}

        {format_nn(output_model['head'], id, 'head')}

        static const uint8_t {id}prob_dot[] = {format_matrix(output_model["prob_dot"])};

        static const uint8_t {id}prob_comma[] = {format_matrix(output_model["prob_comma"])};

        const struct LoremIpsumModel lorem_ipsum_{output_model["lang"]} = {{
            .lang = "{c_str(output_model["lang"])}",
            .name = "{c_str(output_model["name"])}",
            .lower_letters = {id}lower_letters,
            .upper_letters = {id}upper_letters,
            .letters_count = {len(output_model["letters_embedding"])},
            .letters_embedding = {id}letters_embedding,
            .letter_embedding_length = {len(output_model["letters_embedding"][0])},
            .group = {id}group,
            .head = {id}head,
            .prob_dot = {id}prob_dot,
            .prob_comma = {id}prob_comma,
        }};

        '''

    text = dedent(text).strip() + '\n'
    text = re.sub(r'\n\n+', '\n\n', text)
    file.write_text(text)

    languages = [get_lang_config(id) for id in get_languages()]
    languages.sort(key=lambda lang: lang.CODE if lang.CODE != 'la' else '')

    alphabet_definition = {}
    last_language = None
    for lang in languages:
        if last_language is None:
            alphabet_definition[lang.CODE] = (len(lang.ALPHABET), 0, 1, 0 )
        else:
            alphabet_definition[lang.CODE] = (
                f'({len(lang.ALPHABET)} > (_LOREM_IPSUM_{last_language.CODE.upper()}_AS) ? {len(lang.ALPHABET)} : (_LOREM_IPSUM_{last_language.CODE.upper()}_AS))',
                f'_LOREM_IPSUM_{last_language.CODE.upper()}_AS',
                f'_LOREM_IPSUM_{last_language.CODE.upper()}_CN + 1',
                f'_LOREM_IPSUM_{last_language.CODE.upper()}_CN',
            )
        last_language = lang
    
    text = f'''
        #include "lorem-ipsum-int.h"

        #if {" || ".join((f"defined(LOREM_IPSUM_{lang.CODE.upper()}_ENABLED)") for lang in languages)}
        {f"{nl}        ".join((f"#define LOREM_IPSUM_{lang.CODE.upper()}_DISABLED") for lang in languages)}
        #endif

        {"".join((f"""
        #if defined(LOREM_IPSUM_{lang.CODE.upper()}_ENABLED) || !defined(LOREM_IPSUM_{lang.CODE.upper()}_DISABLED)
        #define LOREM_IPSUM_{lang.CODE.upper()}_INSTANCE &lorem_ipsum_{lang.CODE},
        #define _LOREM_IPSUM_{lang.CODE.upper()}_AS {alphabet_definition[lang.CODE][0]}
        #define _LOREM_IPSUM_{lang.CODE.upper()}_CN {alphabet_definition[lang.CODE][2]}
        extern const struct LoremIpsumModel lorem_ipsum_{lang.CODE};
        #else
        #define LOREM_IPSUM_{lang.CODE.upper()}_INSTANCE
        #define _LOREM_IPSUM_{lang.CODE.upper()}_AS {alphabet_definition[lang.CODE][1]}
        #define _LOREM_IPSUM_{lang.CODE.upper()}_CN {alphabet_definition[lang.CODE][3]}
        #endif
        """) for lang in languages)}

        #define LOREM_IPSUM_MODELS {" ".join((f"LOREM_IPSUM_{lang.CODE.upper()}_INSTANCE") for lang in languages)}
        #define LOREM_IPSUM_ALPHABET_MAX_SIZE _LOREM_IPSUM_{last_language.CODE.upper()}_AS
        #define LOREM_IPSUM_MODELS_COUNT (_LOREM_IPSUM_{last_language.CODE.upper()}_CN)
        '''

    text = dedent(text).strip() + '\n'
    text = re.sub(r'\n\n+', '\n\n', text)
    header_file = file.parent.parent / 'include/lorem-ipsum-models.h'
    header_file.write_text(text)
