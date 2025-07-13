
import re
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / 'data/cache/polish-stories.txt'

def download_text():
    from datasets import load_dataset
    ds = load_dataset("JonaszPotoniec/anonimowe-polish-stories")
    text = '\n'.join(row['story'] for row in ds['train'].sort("points", True) if row['points'] >= 100)
    text = re.sub(r'["\'()#+_%$^{}<=>[\]/\\*|&@~-]', ' ', text)
    text = re.sub(r'[^ -\u007FąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', ' ', text)
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
    unsupported_text = re.sub(r"[" + LangPl.ALPHABET + r"., ]", '', text.lower())
    assert unsupported_text == '', f"Unsupported characters found: {unsupported_text[:200]}..."
    return text


class LangPl:
    CODE = 'pl'
    ALPHABET = 'aąbcćdeęfghijklłmnńoóprsśtuwyzźż'

    @staticmethod
    def get_text():
        if CACHE_FILE.exists():
            return CACHE_FILE.read_text()
        text = download_text()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(text)
        return text
