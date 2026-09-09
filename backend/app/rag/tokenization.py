import re
import unicodedata
from threading import Lock, local

import jieba
import snowballstemmer

TECH_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]+|\d+(?:\.\d+)?"
)
CJK = re.compile(r"[\u4e00-\u9fff]+")
DOMAIN_TERMS = ("谱聚类", "深度聚类", "最优传输")
_THREAD_LOCAL = local()
_TOKENIZER_INIT_LOCK = Lock()


def _english_stemmer():
    stemmer = getattr(_THREAD_LOCAL, "english_stemmer", None)
    if stemmer is None:
        stemmer = snowballstemmer.stemmer("english")
        _THREAD_LOCAL.english_stemmer = stemmer
    return stemmer


def _build_cjk_tokenizer():
    with _TOKENIZER_INIT_LOCK:
        tokenizer = jieba.Tokenizer()
        for term in DOMAIN_TERMS:
            tokenizer.add_word(term)
    return tokenizer


_CJK_TOKENIZER = _build_cjk_tokenizer()


def _cjk_tokenizer():
    return _CJK_TOKENIZER


def tokenize_mixed(text: str, *, deduplicate: bool = True) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "")

    tokens: list[str] = []
    for match in TECH_TOKEN.finditer(normalized):
        raw = match.group(0).casefold()
        tokens.append(raw)
        if raw.isalpha() and len(raw) > 3:
            stem = _english_stemmer().stemWord(raw)
            if stem != raw:
                tokens.append(stem)

    for segment in CJK.findall(normalized):
        tokens.extend(
            word.strip() for word in _cjk_tokenizer().lcut(segment) if word.strip()
        )

    return list(dict.fromkeys(tokens)) if deduplicate else tokens


def tokens_to_fts_text(tokens: list[str]) -> str:
    return " ".join(tokens)
