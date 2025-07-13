
import re
from pathlib import Path

def download_text():
    from datasets import load_dataset
    ds = load_dataset("JonaszPotoniec/anonimowe-polish-stories")
    text = '\n'.join(row['story'] for row in ds['train'].sort("points", True) if row['points'] >= 100)
    text = re.sub(r'["\'()#+_%$^{}<=>[\]/\\*|&@~-]', ' ', text)
    text = re.sub(r'[^\u0000-\u007FąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([^\s]*)([a-ząćęłńóśźż])([^a-ząćęłńóśźż\s]+)([a-ząćęłńóśźż]+)", ' ', text, flags=re.IGNORECASE)
    text = re.sub(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,}", ' ', text)
    text = re.sub(r"[0-9]", ' ', text)
    text = re.sub(r"[?!…]", '. ', text)
    text = re.sub(r"[:;]", ', ', text)
    text = text.replace('q', 'k').replace('x', 's').replace('v', 'w')
    text = text.replace('Q', 'K').replace('X', 'S').replace('V', 'W')
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"[\s,.]*\.[\s,.]*", '. ', text)
    text = re.sub(r"[\s,]*,[\s,]*", ', ', text)
    unsupported_text = re.sub(r"[" + LangPl.ALPHABET + r".,]", '', text.lower())
    assert unsupported_text == '', f"Unsupported characters found: {unsupported_text[:200]}..."
    return text


class LangPl:
    LANG = 'pl'
    ALPHABET = ' aąbcćdeęfghijklłmnńoóprsśtuwyzźż'
    ALPHABET_LENGTH = len(ALPHABET) # TODO: This should be outside configuration, it should be derived from ALPHABET somewhere else.
    LETTER_TO_INDEX = {key: value for value, key in enumerate(ALPHABET)}
    INDEX_TO_LETTER = [x for x in ALPHABET]

    @staticmethod
    def get_text():
        cache_file = Path('data/cache/polish-stories.txt')
        if cache_file.exists():
            return cache_file.read_text()
        text = download_text()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
        return text

print(len(LangPl.get_text()))
