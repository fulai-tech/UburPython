"""Stale ES/cache rows cannot authorize absent or changed Mongo materials."""

from unittest.mock import AsyncMock, MagicMock

from app.core.config import Settings
from app.schemas.search_material import project_search_material
from app.server.somni.audio.catalog import AudioCatalogService


class Cursor:
    def __init__(self, rows):
        self.rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.rows)
        except StopIteration:
            raise StopAsyncIteration


async def test_missing_material_is_removed_and_changed_url_is_refreshed_from_mongo():
    current = {
        "_id": "ok",
        "status": True,
        "audio_name": "Rain",
        "audio_url": "https://example.test/current.mp3",
        "sleep_stage_tags": [{"code": "unwind"}],
        "content_form_tags": [{"code": "steady_light_rain"}],
    }
    stale = {**current, "_id": "changed", "audio_url": "https://example.test/old.mp3"}
    retrieval = MagicMock()
    retrieval.search = AsyncMock(
        return_value=[
            project_search_material({**current, "_id": "missing"}),
            project_search_material(stale),
            project_search_material(current),
        ]
    )
    service = AudioCatalogService(None, Settings(), retrieval=retrieval)
    collection = MagicMock()
    collection.find.return_value = Cursor([current, {**current, "_id": "changed"}])
    service._materials = lambda: collection
    result = await service.search_prescription_audio({"sleep_stage_tags": ["unwind"], "top_k": 2})
    assert [x["_id"] for x in result] == ["changed", "ok"]
    assert result[0]["audio_url"] == current["audio_url"]
    assert collection.find.call_args.args[0]["status"] is True
    assert "audio_url" in collection.find.call_args.args[1]
    assert "description_vector" not in collection.find.call_args.args[1]


async def test_no_dislike_answer_is_not_forwarded_as_none_tag():
    from app.schemas.audio import SearchAudioRequest

    assert SearchAudioRequest(disliked_tags=["无", "none", "呼吸引导"]).disliked_tags == [
        "breathing_guidance"
    ]


async def test_authority_snapshot_is_reused_only_within_one_rpc():
    from app.server.somni.audio.catalog import prescription_material_scope

    row = {
        "_id": "one",
        "status": True,
        "audio_url": "https://example.test/one.mp3",
        "sleep_stage_tags": [{"code": "unwind"}],
    }
    service = AudioCatalogService(None, Settings())
    collection = MagicMock()
    collection.find.side_effect = lambda *args: Cursor([row])
    service._materials = lambda: collection
    candidate = project_search_material(row)
    with prescription_material_scope():
        assert await service._verify_prescription_materials([candidate])
        assert await service._verify_prescription_materials([candidate])
        assert collection.find.call_count == 1
    row["status"] = False
    with prescription_material_scope():
        assert await service._verify_prescription_materials([candidate]) == []
    assert collection.find.call_count == 2


async def test_changed_eligibility_cannot_be_reused_from_stale_cache():
    row = {
        "_id": "one",
        "status": True,
        "audio_url": "https://example.test/one.mp3",
        "content_form_tags": [{"code": "breathing_guidance"}],
    }
    service = AudioCatalogService(None, Settings())
    collection = MagicMock()
    collection.find.return_value = Cursor([row])
    service._materials = lambda: collection
    cached = project_search_material({**row, "content_form_tags": [{"code": "steady_light_rain"}]})
    assert await service._verify_prescription_materials([cached]) == []
