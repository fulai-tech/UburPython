"""问卷自由文本分词。

中文用词库切开连写的「小提琴雨声钢琴声」。英文按空格和连字符切开，
丢掉虚词，并把复数、进行时还原成标签英文名里的那个词。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import jieba

# 口头语和连接词。单字在分词结果里本来就会被丢掉。
_FILLERS = frozenset(
    {
        "我不喜欢",
        "不喜欢",
        "喜欢",
        "讨厌",
        "听到",
        "以及",
        "还有",
        "或者",
        "啊",
        "哦",
        "噢",
        "对",
        "就",
        "阿",
        "的",
        "了",
        "和",
        "跟",
        "与",
        "或",
        "是",
        "很",
        "也",
        "在",
        "有",
        "这",
        "那",
        "吗",
        "呢",
        "吧",
    }
)


def cut_zh(text: str) -> list[str]:
    """按词切开，保留原顺序。标点会被分词器丢掉。"""
    raw = text.strip()
    if not raw:
        return []
    return [token.strip() for token in jieba.lcut(raw) if token.strip()]


def segment_dislike_text(text: str) -> list[str]:
    """去掉口头语和单字，得到用于检索的词。"""
    words: list[str] = []
    seen: set[str] = set()
    for token in cut_zh(text):
        if len(token) < 2 or token in _FILLERS or token in seen:
            continue
        seen.add(token)
        words.append(token)
    return words


def tags_overlapping(term: str, names: list[str]) -> list[str]:
    """短词出现在标签名里，或标签名就是这个短词。至少两个字。"""
    if len(term) < 2:
        return []
    return [name for name in names if term in name or name in term]


# 英文虚词。和 NLTK stopwords 的 english 列表一致，避免 and 去对标签英文名。
_EN_STOP = frozenset(
    Path(__file__).with_name("en_stopwords.txt").read_text(encoding="utf-8").split()
)


def _nltk_data_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[4]
    return [root / ".nltk_data", Path("/tmp/nltk_data")]


@lru_cache(maxsize=1)
def _en_lemmatizer():
    """词形还原。词库不在时退回原词，匹配侧还会再去掉词尾 s。"""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return None
    for path in _nltk_data_paths():
        if (path / "corpora" / "wordnet").is_dir():
            nltk.data.path.insert(0, str(path))
            return WordNetLemmatizer()
    return None


def _en_lemma(word: str) -> str:
    token = word.casefold()
    lemmatizer = _en_lemmatizer()
    if lemmatizer is None:
        return token
    noun = lemmatizer.lemmatize(token, "n")
    verb = lemmatizer.lemmatize(token, "v")
    if verb != token and len(verb) >= 3:
        return verb
    return noun


def english_word_forms(word: str) -> set[str]:
    """词本身、词形还原，以及去掉词尾 s 后的形式。humidifiers 词库不认识，靠这一步对上 humidifier。"""
    lemma = _en_lemma(word)
    forms = {lemma}
    if lemma.endswith("s") and len(lemma) > 3:
        forms.add(_en_lemma(lemma[:-1]))
    return {item for item in forms if len(item) >= 3}


def segment_dislike_text_en(text: str) -> list[str]:
    """英文按空格和连字符切开，丢掉虚词和短词。"""
    words: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z]+", text.replace("-", " ")):
        word = _en_lemma(token)
        if len(word) < 3 or word in _EN_STOP or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


@lru_cache(maxsize=512)
def english_name_forms(name: str) -> frozenset[str]:
    """标签英文名里的每个词，含词形还原。"""
    forms: set[str] = set()
    for token in re.findall(r"[A-Za-z]+", name):
        forms |= english_word_forms(token)
    return frozenset(forms)
