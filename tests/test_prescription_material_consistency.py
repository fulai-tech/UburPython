"""Prescription results use ES metadata without Mongo identity refresh."""

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


async def test_es_ids_and_urls_are_not_replaced_or_removed_by_mongo():
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
    retrieval.search_prescription = retrieval.search
    service = AudioCatalogService(None, Settings(), retrieval=retrieval)
    collection = MagicMock()
    collection.find.return_value = Cursor([current, {**current, "_id": "changed"}])
    service._materials = lambda: collection
    result = await service.search_prescription_audio({"sleep_stage_tags": ["unwind"], "top_k": 2})
    assert [x["_id"] for x in result] == ["missing", "changed"]
    assert result[1]["audio_url"] == stale["audio_url"]
    collection.find.assert_not_called()


async def test_no_dislike_answer_is_not_forwarded_as_none_tag():
    from app.schemas.audio import SearchAudioRequest

    assert SearchAudioRequest(disliked_tags=["无", "none", "呼吸引导"]).disliked_tags == [
        "breathing_guidance"
    ]


async def test_repeated_search_does_not_create_mongo_authority_snapshot():
    row = {
        "_id": "one",
        "status": True,
        "audio_url": "https://example.test/one.mp3",
        "sleep_stage_tags": [{"code": "unwind"}],
    }
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[project_search_material(row)])
    retrieval.search_prescription = retrieval.search
    service = AudioCatalogService(None, Settings(), retrieval=retrieval)
    collection = MagicMock()
    collection.find.side_effect = AssertionError("no material Mongo reads")
    service._materials = lambda: collection
    task = {"sleep_stage_tags": ["unwind"], "top_k": 1}
    assert await service.search_prescription_audio(task)
    assert await service.search_prescription_audio(task)
    collection.find.assert_not_called()
    assert retrieval.search.await_count == 2


async def test_retired_reference_rejects_whole_material_with_auditable_warning(monkeypatch):
    from app.server.somni.audio import catalog

    valid = {
        "_id": "good",
        "status": True,
        "audio_url": "https://example.test/good.mp3",
        "content_form_tags": [{"code": "steady_rain"}],
    }
    invalid = {
        **valid,
        "_id": "legacy",
        "content_form_tags": [{"code": "steady_rain"}, {"code": "birdsong"}],
    }
    service = AudioCatalogService(None, Settings())
    collection = MagicMock()
    collection.find.return_value = Cursor([invalid, valid])
    service._materials = lambda: collection
    warning = MagicMock()
    monkeypatch.setattr(catalog.logger, "warning", warning)
    result = service._filter_prescription_materials(
        [project_search_material(invalid), project_search_material(valid)]
    )
    assert [row["_id"] for row in result] == ["good"]
    args = warning.call_args.args
    assert args[1] == "legacy" and "birdsong" in args[3]
    assert "https://" not in repr(args)
