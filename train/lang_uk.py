
import re
from pathlib import Path
from tqdm import tqdm

CACHE_FILE = Path(__file__).parent.parent / 'data/cache/allenai-c4-uk.txt'
TEXT_SIZE_LIMIT = 20 * 1024 * 1024

def download_text():
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "uk", streaming=True)
    subset_rows = []
    subset_size = 0
    progress_bar = tqdm(total=TEXT_SIZE_LIMIT)
    for row in ds['train']:
        subset_rows.append(row['text'])
        subset_size += len(row['text'])
        if subset_size >= TEXT_SIZE_LIMIT:
            progress_bar.clear()
            break
        progress_bar.update(len(row['text']))
    text = ' '.join(subset_rows)
    a = LangUk.ALPHABET + LangUk.ALPHABET.upper()
    upper = LangUk.ALPHABET.upper()
    text = re.sub(r'[\x01- –«»"\'()#+_%$^{}<=>[\]/\\*|&@~-]', ' ', text)
    text = re.sub(rf'[^ -\u007F{a}]', ' ', text)
    text = re.sub(r'[\s.,:?]+[a-zA-Z]+', ' ', text)
    text = re.sub(r'[\s\u00A0]+', ' ', text)
    text = re.sub(rf"([^\s]*)([{a}])([^{a}\s]+)([{a}]+)", ' ', text)
    text = re.sub(rf"[{upper}][{upper}]+", ' ', text)
    for i in range(3):
        text = re.sub(r"(Р|С|РїР|РєР|Рі|Рґ|Рє|Рї)\s", ' ', text)
    text = re.sub(r"[0-9]", ' ', text)
    text = re.sub(r"[?!…]", '. ', text)
    text = re.sub(r"[:;`]", ', ', text)
    text = re.sub(r'[a-zA-Z]+', '', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"[\s,.]*\.[\s,.]*", '. ', text)
    text = re.sub(r"[\s,]*,[\s,]*", ', ', text)
    unsupported_text = re.sub(r"[" + a + r"., ]", '', text.lower())
    assert unsupported_text == '', f"Unsupported characters found: {unsupported_text[:200]}..."
    return text

class LangUk:

    NAME = 'Ukrainian'

    ALPHABET = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'

    @staticmethod
    def get_text():
        if CACHE_FILE.exists():
            return CACHE_FILE.read_text()
        text = download_text()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(text)
        return text
