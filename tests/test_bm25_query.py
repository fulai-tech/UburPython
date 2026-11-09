"""BM25 长短句 phrase 拆分。"""

from app.server.somni.audio.bm25_query import build_bm25_must_clause, extract_bm25_phrases


def test_short_zh_keeps_full_phrase() -> None:
    assert extract_bm25_phrases("雨声") == ["雨声"]
    assert build_bm25_must_clause("description_text", "雨声") == {
        "match_phrase": {"description_text": "雨声"}
    }


def test_short_en_keeps_full_phrase() -> None:
    assert extract_bm25_phrases("white noise") == ["white noise"]
    assert extract_bm25_phrases("rain") == ["rain"]


def test_long_en_splits_content_words() -> None:
    phrases = extract_bm25_phrases("soft piano for bedtime")
    assert "piano" in phrases
    assert "bedtime" in phrases
    assert "for" not in phrases
    clause = build_bm25_must_clause("description_text_en", "soft piano for bedtime")
    assert "bool" in clause
    assert clause["bool"]["minimum_should_match"] == 1


def test_multi_color_zh_splits_aliases() -> None:
    phrases = extract_bm25_phrases("白噪音和粉噪音")
    assert "白噪音" in phrases
    assert "粉噪音" in phrases
    assert "白噪" not in phrases
    assert "噪音" not in phrases


def test_long_zh_uses_tokens() -> None:
    phrases = extract_bm25_phrases("下雨的声音")
    assert phrases
    assert "的" not in phrases
