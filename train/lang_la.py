
import re
from pathlib import Path

LOREM_IPSUM_ORIGINAL = '''
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt
in culpa qui officia deserunt mollit anim id est laborum.
'''.strip()

CACHE_FILE = Path(__file__).parent.parent / 'data/cache/ancient-latin-passages.txt'
RAW_CACHE_FILE = CACHE_FILE.with_suffix('.raw.txt')

def download_text():
    if not RAW_CACHE_FILE.exists():
        import pandas as pd
        df = pd.read_json("hf://datasets/lsb/ancient-latin-passages/ancient-latin-passages.json", lines=True)
        texts = [LOREM_IPSUM_ORIGINAL]
        for _, row in df.iterrows():
            for col in df.columns:
                texts.append(row[col])
        text = '.\n'.join(texts)
        RAW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        RAW_CACHE_FILE.write_text(text)
    else:
        text = RAW_CACHE_FILE.read_text()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([^a-zA-Z])[A-Z]+([^a-zA-Z])", "\\1 \\2", text)
    text = re.sub(r'[^a-z,.\s]+', '.', text, flags=re.IGNORECASE)
    text = re.sub(r'[\s,.]*\.[\s,.]*', '. ', text)
    text = re.sub(r'[\s,]*,[\s,]*', ', ', text)
    text = text.replace('w', 'v').replace('W', 'V') # "W" did not exist at all.
    text = text.replace('j', 'i').replace('J', 'I') # "I" was used for both I and J sounds.
    text = text.replace('k', 'g').replace('K', 'G') # K, Y, Z were rarely used in native Latin words.
    text = text.replace('y', 'i').replace('Y', 'I')
    text = text.replace('z', 's').replace('Z', 'S')
    return text


class LangLa:
    CODE = 'la'
    ALPHABET = "abcdefghilmnopqrstuvx"

    @staticmethod
    def get_text():
        if CACHE_FILE.exists():
            return CACHE_FILE.read_text()
        text = download_text()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(text)
        return text
