"""问卷自由文本分词。

中文用词库切开连写的「小提琴雨声钢琴声」。英文按空格和连字符切开，
丢掉虚词，并把复数、进行时还原成标签英文名里的那个词。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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


# 「不喜欢」整句里的连接词：在这里把一句话切成几个短语
_PHRASE_BREAKS = frozenset({"和", "与", "跟", "或", "或者", "以及", "还有", "及"})

# 短语首尾的口头语、否定和程度词，去掉后剩下的才是声音本身
_PHRASE_EDGE_FILLERS = _FILLERS | {
    "我",
    "也",
    "都",
    "真的",
    "特别",
    "非常",
    "比较",
    "有点",
    "不太",
    "不要",
    "不想",
    "不想听",
    "不想要",
    "不爱",
    "不爱听",
    "听",
    "怕",
    "害怕",
    "我怕",
    "受不了",
    "别",
    "别放",
    "避免",
    "吵",
    "太吵",
    "太",
    "声音",
    "音频",
}

# 只说「声音」「音频」这类泛称时，不拿去匹配
_GENERIC_SOUND_WORDS = frozenset({"声音", "声响", "音频", "东西"})

# 只用默认词典，不往里加词。标签名在每次请求里现场定位，标签改名或下线立即生效。
_phrase_tokenizer = jieba.Tokenizer()


def _is_break_token(token: str) -> bool:
    return token in _PHRASE_BREAKS or not re.search(r"\w", token)


def _tokens_keeping_names(text: str, vocabulary: Iterable[str]) -> list[str]:
    """默认分词，再把原句里完整出现的标签名合并成一个词。

    标签名两端都要落在分词边界上，「流水声」里的「水声」不算。长的优先，互不重叠。
    """
    pieces = list(_phrase_tokenizer.tokenize(text))
    bounds = {start for _, start, _ in pieces} | {end for _, _, end in pieces}
    taken = [False] * len(text)
    spans: dict[int, int] = {}
    for name in sorted({word for word in vocabulary if len(word) >= 2}, key=len, reverse=True):
        start = text.find(name)
        while start != -1:
            end = start + len(name)
            if start in bounds and end in bounds and not any(taken[start:end]):
                spans[start] = end
                taken[start:end] = [True] * len(name)
            start = text.find(name, start + 1)
    tokens: list[str] = []
    for word, start, _ in pieces:
        if start in spans:
            tokens.append(text[start : spans[start]])
        elif not taken[start]:
            tokens.append(word)
    return tokens


def segment_dislike_phrases(text: str, vocabulary: Iterable[str] = ()) -> list[str]:
    """按连接词和标点切成短语，去掉首尾口头语。

    vocabulary 是标签名，原句里完整出现的标签名当作一个词，「白噪音」「人声引导」不会被切碎。
    四个字以内的短语整体保留；若它全由多字词组成（湖水拍打），再补上这些词。
    切出单字的短语是复合词（滴水声→滴/水声），拆开会丢掉修饰，不补。
    更长的短语只取多字词。
    """
    phrases: list[list[str]] = [[]]
    for token in _tokens_keeping_names(text.strip(), vocabulary):
        token = token.strip()
        if not token or _is_break_token(token):
            phrases.append([])
            continue
        phrases[-1].append(token)
    words: list[str] = []
    for tokens in phrases:
        while tokens and tokens[0] in _PHRASE_EDGE_FILLERS:
            tokens = tokens[1:]
        while tokens and tokens[-1] in _PHRASE_EDGE_FILLERS:
            tokens = tokens[:-1]
        if not tokens:
            continue
        inner = [
            token
            for token in tokens
            if len(token) >= 2 and token not in _PHRASE_EDGE_FILLERS
        ]
        phrase = "".join(tokens)
        if len(phrase) > 4:
            words.extend(inner)
            continue
        words.append(phrase)
        if all(len(token) >= 2 for token in tokens):
            words.extend(inner)
    return [
        word
        for word in dict.fromkeys(words)
        if len(word) >= 2 and word not in _GENERIC_SOUND_WORDS
    ]


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
