"""Vocal identity from enabled content tags, separate from engineering values."""

from collections.abc import Mapping
from typing import Any

from app.core.audio_tags import dictionary_records, validated_content_references

VOICE_CONTENT_CODES = frozenset(
    {
        "voice",
        "lyrical_song",
        "non_lexical_vocal",
        "asmr_whisper",
        "low_plot_sleep_story",
        "voice_guidance",
        "wake_voice",
        "breathing_guidance",
        "body_scan",
        "companion_presence_guidance",
        "guided_imagery",
        "loving_kindness_meditation",
        "mindfulness_meditation",
        "mindfulness_sleep",
        "morning_stretch_guidance",
        "morning_orientation",
        "night_awakening_guidance",
        "nightmare_awakening_guidance",
        "progressive_muscle_relaxation",
    }
)


def content_voice_state(material: Mapping[str, Any]) -> bool | None:
    """Return unknown for absent/invalid refs, never classify using title prose."""
    try:
        codes = validated_content_references(material)
    except ValueError:
        return None
    if not codes:
        return None
    rows = dictionary_records()
    ids = {row["id"]: row["code"] for row in rows}
    parents = {row["code"]: ids.get(row.get("parent_tag_id"), "") for row in rows}
    for code in codes:
        seen = set()
        while code and code not in seen:
            if code in VOICE_CONTENT_CODES:
                return True
            seen.add(code)
            code = parents.get(code, "")
    return False
