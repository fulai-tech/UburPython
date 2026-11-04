"""GetAudio BM25 查询构造：短词整句 phrase；长句拆成关键词 phrase。"""

from __future__ import annotations

import re
from functools import lru_cache

from app.server.somni.audio.relevance_filter import _COLOR_ALIAS_KEYS

_EN_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "for",
        "to",
        "of",
        "and",
        "or",
        "with",
        "without",
        "in",
        "on",
        "at",
        "is",
        "are",
        "be",
        "good",
        "best",
        "some",
        "any",
        "my",
        "me",
        "i",
        "please",
        "want",
        "like",
        "sounds",
        "sound",
        "audio",
    }
)
_ZH_STOP = frozenset(
    {
        "的",
        "了",
        "吗",
        "呢",
        "吧",
        "啊",
        "我",
        "想",
        "要",
        "有",
        "没有",
        "帮",
        "给我",
        "一下",
        "什么",
        "怎么",
        "如何",
        "适合",
        "听听",
        "听",
        "声音",
        "音频",
    }
)
_ZH_SHORT_MAX = 4
_EN_SHORT_MAX_TOKENS = 2


def extract_bm25_phrases(query_text: str) -> list[str]:
    """短查询返回原句；长查询返回色噪音别名 + 实词，供 should match_phrase。"""
    text = (query_text or "").strip()
    if not text:
        return []

    color_phrases = _dedupe_color_phrases(_color_phrases_in_text(text))
    if _is_english_query(text):
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        content = [
            t
            for t in tokens
            if t not in _EN_STOP
            and len(t) >= 3
            and not _covered_by_phrase(t, color_phrases)
        ]
        if len(tokens) <= _EN_SHORT_MAX_TOKENS and not _needs_split(text, color_phrases):
            return [text]
        phrases = list(dict.fromkeys([*color_phrases, *content]))
        return phrases or [text]

    if len(text) <= _ZH_SHORT_MAX and not _needs_split(text, color_phrases):
        return [text]
    content = [
        token
        for token in _zh_content_tokens(text)
        if token not in {"噪音", "噪声", "声音"}
        and not _covered_by_phrase(token, color_phrases)
    ]
    phrases = list(dict.fromkeys([*color_phrases, *content]))
    return phrases or [text]


def build_bm25_must_clause(text_field: str, query_text: str) -> dict[str, object]:
    """生成 bool.must 里的单条子句（match_phrase 或 should 多 phrase）。"""
    phrases = extract_bm25_phrases(query_text)
    if len(phrases) <= 1:
        phrase = phrases[0] if phrases else (query_text or "").strip()
        return {"match_phrase": {text_field: phrase}}
    return {
        "bool": {
            "should": [{"match_phrase": {text_field: phrase}} for phrase in phrases],
            "minimum_should_match": 1,
        }
    }


def _needs_split(text: str, color_phrases: list[str]) -> bool:
    """句中出现多个色噪音，或明显并列时强制拆开。"""
    if len(color_phrases) >= 2:
        return True
    return bool(re.search(r"[和与跟或,/，、]| and | or ", text, flags=re.IGNORECASE))


def _color_phrases_in_text(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for alias in _COLOR_ALIAS_KEYS:
        if alias in lowered and alias not in found:
            found.append(alias)
    return found


def _dedupe_color_phrases(phrases: list[str]) -> list[str]:
    """长别名优先；已有「白噪音」则丢掉「白噪」。"""
    kept: list[str] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        if any(phrase != other and phrase in other for other in kept):
            continue
        kept.append(phrase)
    return kept


def _covered_by_phrase(token: str, phrases: list[str]) -> bool:
    needle = token.casefold()
    return any(needle in phrase.casefold() for phrase in phrases)


def _is_english_query(text: str) -> bool:
    return any("A" <= c <= "Z" or "a" <= c <= "z" for c in text) and not any(
        "\u4e00" <= c <= "\u9fff" for c in text
    )


@lru_cache(maxsize=1)
def _jieba_tokenizer():
    import jieba

    return jieba.Tokenizer()


def _zh_content_tokens(text: str) -> list[str]:
    tokenizer = _jieba_tokenizer()
    return [
        token.strip()
        for token, _start, _end in tokenizer.tokenize(text)
        if len(token.strip()) >= 2 and token.strip() not in _ZH_STOP
    ]
