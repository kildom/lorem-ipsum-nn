
import re
import json
import textwrap
from pathlib import Path
from textwrap import dedent
from lang import get_lang_config, get_languages

MAX_NUMBER_ARRAY_WIDTH = 100

def format_value(value, indent, output_model: dict, key=''):
    if isinstance(value, dict):
        result = [f'{{\n']
        for key, val in value.items():
            result.append(f'{indent}    {key}: {format_value(val, indent + "    ", output_model, key)},\n')
        result.append(f'{indent}}}')
        return ''.join(result)
    elif isinstance(value, list):
        only_numbers = all(isinstance(item, (int, float)) for item in value)
        if only_numbers:
            assert all(round(item) == item for item in value)
            text = ', '.join(f'{int(item)}' for item in value)
            lines = list(textwrap.wrap(text, width=MAX_NUMBER_ARRAY_WIDTH))
            if len(lines) == 1:
                return '[' + text + ']'
            return '[\n' + '\n'.join(f'{indent}    {line}' for line in lines) + f'\n{indent}]'
        else:
            result = [f'[\n']
            for i, item in enumerate(value):
                result.append(f'{indent}    {format_value(item, indent + "    ", output_model)},')
                if key == 'letters_embedding':
                    space = ' ' * (24 - len(result[-1]))
                    result.append(f'{space}// {output_model["letters"][i]}'.rstrip())
                result.append('\n')
            result.append(f'{indent}]')
            return ''.join(result)
    elif isinstance(value, (str, int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    else:
        raise TypeError(f'Unsupported type: {type(value)}')


def format_ts(output_model: dict, file: Path):

    file = Path(__file__).parent.parent / ('src/ts/' + file.with_suffix('.ts').name)

    text = f'\nexport const {output_model["lang"]}Model = {format_value(output_model, "", output_model)};\n'

    file.write_text(text)

    languages = [get_lang_config(id) for id in get_languages()]
    languages.sort(key=lambda lang: lang.CODE if lang.CODE != 'la' else '')

    text = '\n'
    for lang in languages:
        text += f'import {{ {lang.CODE}Model }} from "./{lang.CODE}";\n'
    text += '\n'
    text += f'export const models: {{ [key: string]: typeof {languages[0].CODE}Model }} = {{\n'
    for lang in languages:
        text += f'    {lang.CODE}: {lang.CODE}Model,\n'
    text += '};\n\n'
    text += f'export {{ {languages[0].CODE}Model as defaultModel }};\n'
    header_file = file.parent / 'models.ts'
    header_file.write_text(text)
