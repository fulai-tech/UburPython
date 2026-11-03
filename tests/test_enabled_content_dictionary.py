"""Prevent legitimate database content tags from being lost in a smaller snapshot."""

import json
import shutil
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.audio_tags import (
    dictionary_records,
    validated_audio_tags,
    validated_content_references,
)
from app.core.audio_voice_policy import content_voice_state
from app.core.config import Settings
from app.server.somni.audio.catalog import AudioCatalogService
from scripts.export_audio_tag_contract import CONTENT_FILTER, normalize_rows


@pytest.mark.parametrize("row", dictionary_records(), ids=lambda row: row["code"])
def test_every_enabled_content_code_is_admitted(row):
    assert row["type"] == "content_form" and row["status"] == "启用"
    assert validated_audio_tags([row["code"]]) == [row["code"]]


@pytest.mark.parametrize("kind", ["content_form", "mechanism", "audio_engineering", "sleep_stage"])
@pytest.mark.parametrize("status", ["启用", "禁用", True, None])
def test_export_filter_is_exact(kind, status):
    rows = normalize_rows([{"_id": "one", "code": "one", "type": kind, "status": status}])
    assert len(rows) == int(kind == "content_form" and status == "启用")
    assert CONTENT_FILTER == {"type": "content_form", "status": "启用"}


@pytest.mark.parametrize(
    "code", ["loopable", "voice_position", "breathing_pacing", "wake", "unknown"]
)
def test_wrong_dimensions_and_unknown_tags_are_not_content(code):
    with pytest.raises(ValueError):
        validated_audio_tags([code])
    with pytest.raises(ValueError):
        validated_content_references({"content_form_tags": [{"code": code}]})


@pytest.mark.parametrize("phase", ["unwind", "soothe", "guard", "wake"])
def test_protocol_stages_are_independent_of_content_dictionary(phase):
    assert validated_audio_tags([phase], allowed_types=frozenset({"sleep_stage"})) == [phase]


@pytest.mark.parametrize(
    "code", ["body_scan", "paradoxical_intention", "postponement", "voice_guidance"]
)
def test_current_vocal_tags_do_not_need_retired_engineering_voice_position(code):
    assert content_voice_state({"content_form_tags": [{"code": code}]}) is True
    assert content_voice_state({"content_form_tags": [{"code": "steady_light_rain"}]}) is False
    assert content_voice_state({"content_form_tags": [{"code": "unreviewed"}]}) is None


@pytest.mark.asyncio
async def test_all_regression_codes_reach_real_catalog_validation():
    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def __aiter__(self):
            self.iterator = iter(self.rows)
            return self

        async def __anext__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                raise StopAsyncIteration from None

    rows = dictionary_records()
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[])
    catalog = AudioCatalogService(None, Settings(), retrieval=retrieval)
    collection = MagicMock()
    collection.find.side_effect = lambda query, projection: Cursor(
        [row for row in rows if row["code"] in query["code"]["$in"]]
    )
    catalog._tags = lambda: collection
    codes = [
        "deep_slow_river",
        "harp_music",
        "imagery_distraction",
        "mindfulness_meditation",
        "postponement",
        "singing_bowl",
        "wind_ambience",
        "wuxing_healing_music",
    ]
    assert (
        await catalog.search_prescription_audio(
            {"content_tag_codes": codes, "sleep_stage_tags": ["unwind"], "top_k": 1}
        )
        == []
    )
    assert set(retrieval.search.await_args.args[0].content_tag_codes) == set(codes)
    assert collection.find.call_args.args[0]["type"] == "content_form"
    assert collection.find.call_args.args[0]["status"] == "启用"


def test_live_drift_check_rejects_missing_rows_without_rewriting_assets(tmp_path, monkeypatch):
    from scripts import export_audio_tag_contract as exporter

    rows = dictionary_records()
    path = tmp_path / "audio_tag_aliases.json"
    original = json.dumps({"dictionary": rows})
    path.write_text(original)
    historical = tmp_path / "audio_tag_aliases_v5.json"
    historical.write_text("frozen-history")
    monkeypatch.setattr(exporter, "CORE", tmp_path)
    monkeypatch.setattr(exporter, "read_dictionary", lambda env_file: rows[:-1])
    with pytest.raises(ValueError, match="audio_live_dictionary_drift"):
        exporter.check_live(tmp_path / "operator.env")
    assert path.read_text() == original
    assert historical.read_text() == "frozen-history"


def test_sync_refuses_to_overwrite_existing_historical_mirror(tmp_path):
    from scripts import export_audio_tag_contract as exporter

    mirror = tmp_path / "mirror"
    assets = mirror / "conf/somni"
    source = mirror / "src/utils"
    assets.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "audio_tags.py").write_text(
        exporter.dph_module((exporter.CORE / "audio_tags.py").read_text())
    )
    for name in (
        "audio_tag_aliases.json",
        "audio_preference_rules.json",
        "audio_tag_aliases_v2.json",
        "audio_tag_aliases_v5.json",
    ):
        shutil.copyfile(exporter.CORE / name, assets / name)
    frozen = assets / "audio_tag_aliases_v5.json"
    frozen.write_text('{"frozen":"different-history"}')
    current = (assets / "audio_tag_aliases.json").read_bytes()
    with pytest.raises(ValueError, match="audio_historical_dictionary_mirror_drift"):
        exporter.synchronize(mirror, write=True)
    assert frozen.read_text() == '{"frozen":"different-history"}'
    assert (assets / "audio_tag_aliases.json").read_bytes() == current
