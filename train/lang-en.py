import re
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / 'data/cache/onestop-english.txt'

def download_text():
    from datasets import load_dataset
    ds = load_dataset("iastate/onestop_english")
    text = '\n'.join(row['text'] for row in ds['train'].sort("label", True))
    text = re.sub(r'[\'’][a-z]([^a-z])', '\\1', text, flags=re.IGNORECASE)
    text = re.sub(r'["()#+_%$^{}<=>[\]/\\*|&@~\'’-]', ' ', text)
    text = re.sub(r'[^ -\u007F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([^\s]*)([a-z])([^a-z\s]+)([a-z]+)", ' ', text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Z]{2,}", ' ', text)
    text = re.sub(r"[0-9]", ' ', text)
    text = re.sub(r"[?!…]", '. ', text)
    text = re.sub(r"[:;]", ', ', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"[\s,.]*\.[\s,.]*", '. ', text)
    text = re.sub(r"[\s,]*,[\s,]*", ', ', text)
    unsupported_text = re.sub(r"[" + LangEn.ALPHABET + r"., ]", '', text.lower())
    assert unsupported_text == '', f"Unsupported characters found: '{unsupported_text[:200]}'"
    return text


class LangEn:

    # Language code
    CODE = 'en'

    # Language alphabet, generated text will also use space, comma, and period that are not included here.
    ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

    # Returns a training text. The string must contains only characters from
    # the alphabet (lower or upper case), space, comma, and period.
    @staticmethod
    def get_text():
        if CACHE_FILE.exists():
            return CACHE_FILE.read_text()
        text = download_text()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(text)
        return text
