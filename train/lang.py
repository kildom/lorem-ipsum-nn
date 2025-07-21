import importlib.util
from pathlib import Path


class LangConfig:
    NAME: str
    CODE: str
    ALPHABET: str
    ALPHABET_LENGTH: int
    INDEX_TO_LETTER: list[str]
    LETTER_TO_INDEX: dict[str, int]

    def __init__(self, parent, language_code):
        self.parent = parent
        self.CODE = language_code
        self.NAME = parent.NAME
        self.ALPHABET = ' ' + parent.ALPHABET
        self.ALPHABET_LENGTH = len(self.ALPHABET)
        self.INDEX_TO_LETTER = [i for i in self.ALPHABET]
        self.LETTER_TO_INDEX = {letter: index for index, letter in enumerate(self.ALPHABET)}

    def get_text(self) -> str:
        return self.parent.get_text()


def get_languages() -> list[str]:
    directory = Path(__file__).parent
    lang_files = directory.glob("lang_*.py")

    languages = []
    for file in lang_files:
        lang_code = file.stem.replace("lang_", "")
        languages.append(lang_code)

    return sorted(languages)


def get_lang_config(language_code: str) -> LangConfig:
    filepath = Path(__file__).parent / f"lang_{language_code}.py"

    if not filepath.is_file():
        raise FileNotFoundError(f"No such language configuration file: {filepath}")

    spec = importlib.util.spec_from_file_location(filepath.stem, str(filepath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        if attr_name.startswith("Lang"):
            attr = getattr(module, attr_name)
            if isinstance(attr, type):
                return LangConfig(attr, language_code)

    raise ImportError(f"No class starting with 'Lang' found in {filepath.name}")
