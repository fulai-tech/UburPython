from app.server.somni.audio.segment import (
    cut_zh,
    segment_dislike_phrases,
    segment_dislike_text,
    segment_dislike_text_en,
    tags_overlapping,
)


def test_dislike_phrases_keep_tag_names_whole() -> None:
    vocabulary = ["白噪音", "人声引导"]
    assert segment_dislike_phrases("我不喜欢海浪声和雨声以及白噪音", vocabulary) == [
        "海浪声",
        "雨声",
        "白噪音",
    ]
    assert segment_dislike_phrases("不喜欢人声引导和冥想", vocabulary) == ["人声引导", "冥想"]


def test_dislike_phrases_strip_fillers_and_generic_words() -> None:
    assert segment_dislike_phrases("讨厌鸟叫、风声，还有空调的声音") == ["鸟叫", "风声", "空调"]
    assert segment_dislike_phrases("大提琴和小提琴都不喜欢") == ["大提琴", "小提琴"]
    assert segment_dislike_phrases("八音盒的声音太吵") == ["八音盒"]


def test_dislike_phrases_short_phrase_also_yields_inner_words() -> None:
    assert segment_dislike_phrases("讨厌湖水拍打的声音") == ["湖水拍打", "湖水", "拍打"]
    assert segment_dislike_phrases("颂钵和合成器") == ["颂钵", "合成器"]


def test_dislike_phrases_compound_with_single_char_stays_whole() -> None:
    assert segment_dislike_phrases("洒水声和滴水声") == ["洒水声", "滴水声"]


def test_dislike_phrases_tag_name_must_sit_on_word_boundary() -> None:
    assert segment_dislike_phrases("不喜欢流水声", ["水声"]) == ["流水声"]


def test_dislike_phrases_vocabulary_does_not_leak_between_calls() -> None:
    assert segment_dislike_phrases("不喜欢人声引导和冥想", ["人声引导"]) == ["人声引导", "冥想"]
    assert segment_dislike_phrases("不喜欢人声引导和冥想") == ["人声", "引导", "冥想"]


def test_dislike_phrases_long_phrase_only_inner_words() -> None:
    assert segment_dislike_phrases("小提琴雨声钢琴声") == ["小提琴", "雨声", "钢琴"]
    assert segment_dislike_phrases("不喜欢水下的气泡声") == ["水下", "气泡"]


def test_joined_sounds_split_into_words() -> None:
    assert cut_zh("小提琴雨声钢琴声") == ["小提琴", "雨声", "钢琴", "声"]
    assert segment_dislike_text("小提琴雨声钢琴声") == ["小提琴", "雨声", "钢琴"]


def test_sentence_with_connectors_drops_fillers() -> None:
    assert segment_dislike_text("我不喜欢小提琴和雨声以及钢琴声") == ["小提琴", "雨声", "钢琴"]


def test_punctuation_list_keeps_each_sound() -> None:
    assert segment_dislike_text("小提琴、女性人声、雨声") == ["小提琴", "女性", "人声", "雨声"]


def test_dislike_prefix_is_removed() -> None:
    assert segment_dislike_text("讨厌雨声") == ["雨声"]


def test_nonsense_keeps_dictionary_words() -> None:
    assert cut_zh("对哦温度计噢噢啊减速电机阿就死哦就") == [
        "对",
        "哦",
        "温度计",
        "噢",
        "噢",
        "啊",
        "减速",
        "电机",
        "阿",
        "就",
        "死哦",
        "就",
    ]
    assert segment_dislike_text("对哦温度计噢噢啊减速电机阿就死哦就") == [
        "温度计",
        "减速",
        "电机",
        "死哦",
    ]


def test_empty_text() -> None:
    assert cut_zh("  ") == []
    assert segment_dislike_text("") == []


def test_english_dislike_drops_function_words_and_lemmatizes() -> None:
    assert segment_dislike_text_en("Keep out thunder, barking, and sirens.") == [
        "keep",
        "thunder",
        "bark",
        "siren",
    ]
    assert segment_dislike_text_en("no more clock ticking please") == ["clock", "tick", "please"]
    assert segment_dislike_text_en("white-noise fans") == ["white", "noise", "fan"]
    assert segment_dislike_text_en("ocean-waves plus campfire crackle") == [
        "ocean",
        "wave",
        "plus",
        "campfire",
        "crackle",
    ]
    assert segment_dislike_text_en("asdfgh qwertyuiop zzzzz") == ["asdfgh", "qwertyuiop", "zzzzz"]
    assert segment_dislike_text_en("小提琴雨声钢琴声") == []


def test_tag_overlap_requires_two_characters() -> None:
    names = ["场景雨声", "中雨/稳定雨声", "慢钢琴", "小提琴"]
    assert tags_overlapping("雨声", names) == ["场景雨声", "中雨/稳定雨声"]
    assert tags_overlapping("钢琴", names) == ["慢钢琴"]
    assert tags_overlapping("声", names) == []
