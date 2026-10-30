"""Current Mongo material contract and DeepHealth prescription RPC regression."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.core.audio_tags import material_is_excluded, validated_audio_tags
from app.core.config import Settings
from app.schemas.search_material import project_search_material
from app.server.somni.audio.catalog import AudioCatalogService
from app.server.somni.audio.rpc import AudioRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc


class Cursor:
    def __init__(self, rows):
        self.rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.rows)
        except StopIteration:
            raise StopAsyncIteration from None


def material():
    return {
        "_id": "current-audio",
        "status": True,
        "audio_name": "Current material",
        "audio_url": "https://example.test/audio.mp3",
        "sleep_stage_tags": [{"code": "unwind", "tag_id": "stage-id"}],
        "content_form_tags": [{"code": "worry_postponement", "tag_id": "content-id"}],
        "audio_engineering_tags": [
            {"code": "voice_position", "value": {"code": "none", "name": "无"}},
            {"code": "loopable", "value": {"code": "conditional"}},
            {
                "code": "spectral_profile",
                "value": None,
                "band_values": [{"code": "low", "value": 0.2}],
                "relative_loudness": -12.5,
            },
        ],
        "duration_seconds": 1800,
        "acoustic_metrics": {
            "duration_seconds": 1800,
            "tempo_bpm": None,
            "transient_ratio": 0.1,
            "transient_max_delta_db": 12,
            "voice_presence_ratio": 0.0,
        },
        "languages": ["zh"],
        "source": {"name": "licensed"},
    }


@pytest.mark.parametrize(
    "code",
    [
        "worry_postponement",
        "birdsong",
        "fan_sound",
        "forest_night",
        "rain_on_window",
        "thunder",
        "gradual_wake_music",
        "melodic_alarm_music",
    ],
)
def test_current_content_codes_are_valid(code):
    assert validated_audio_tags([code], allowed_types=frozenset({"content_form"})) == [code]


def test_wrong_dimension_and_unknown_codes_remain_invalid():
    for code in ["loopable", "unknown_content_code", "85"]:
        with pytest.raises(ValueError, match="unknown_audio_tag"):
            validated_audio_tags([code], allowed_types=frozenset({"content_form"}))


def test_full_material_preserves_optional_metadata_and_null_bpm():
    row = material()
    projected = project_search_material(row)
    for field in [
        "content_form_tags",
        "audio_engineering_tags",
        "acoustic_metrics",
        "duration_seconds",
        "languages",
        "source",
    ]:
        assert projected[field] == row[field]
    projected["audio_engineering_tags"][0]["value"]["code"] = "continuous"
    assert row["audio_engineering_tags"][0]["value"]["code"] == "none"


def test_sparse_material_does_not_invent_metrics():
    projected = project_search_material({"_id": "sparse"})
    assert "duration_seconds" not in projected
    assert "acoustic_metrics" not in projected


@pytest.mark.parametrize("status", [False, "false", "true", 1, None])
def test_only_boolean_true_is_enabled(status):
    assert project_search_material({"status": status})["status"] is False


def test_retired_fields_do_not_change_material_classification():
    row = material()
    row["mechanism_tags"] = [{"code": "birdsong"}]
    row["medical_risk_tags"] = [{"code": "birdsong"}]
    assert not material_is_excluded(row, ["birdsong"])
    assert not {"mechanism_tags", "medical_risk_tags"} & project_search_material(row).keys()
    assert not material_is_excluded(row, ["voice"])
    row["audio_engineering_tags"][0]["value"]["code"] = "continuous"
    assert material_is_excluded(row, ["voice"])


async def test_current_tag_batch_reaches_retrieval_and_preserves_struct_metadata():
    row = material()
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[project_search_material(row)])
    service = AudioCatalogService(None, Settings(), retrieval=retrieval)
    materials = MagicMock()
    materials.find.side_effect = lambda *args: Cursor([row])
    tags = MagicMock()
    tags.find.side_effect = lambda *args: Cursor(
        [{"code": "worry_postponement", "type": "content_form", "name": "担忧延后"}]
    )
    service._materials = lambda: materials
    service._tags = lambda: tags
    request = uburnode_somni_pb2.SearchPrescriptionAudioReq(
        tasks=[
            uburnode_somni_pb2.PrescriptionAudioSearchTask(
                task_id=f"task-{index}",
                sleep_stage_tags=["unwind"],
                content_tag_codes=["worry_postponement"],
                disliked_tags=["voice"],
                top_k=1,
            )
            for index in range(34)
        ]
    )
    server = grpc.aio.server()
    uburnode_somni_pb2_grpc.add_AudioServiceServicer_to_server(AudioRpc(service), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = uburnode_somni_pb2_grpc.AudioServiceStub(channel)
            result = await stub.SearchPrescriptionAudio(request, timeout=5)
    finally:
        await server.stop(None)
    assert len(result.results) == 34
    for item in result.results:
        candidate = item.candidates[0]
        assert candidate["duration_seconds"] == 1800
        assert candidate["acoustic_metrics"]["tempo_bpm"] is None
        assert candidate["audio_engineering_tags"][1]["value"]["code"] == "conditional"
    assert materials.find.call_count == 1
    assert retrieval.search.await_count == 34


async def test_stale_engineering_or_disabled_material_is_not_authorized():
    row = material()
    service = AudioCatalogService(None, Settings())
    collection = MagicMock()
    collection.find.side_effect = lambda *args: Cursor([row] if row["status"] else [])
    service._materials = lambda: collection
    stale = deepcopy(row)
    stale["audio_engineering_tags"][1]["value"]["code"] = "true"
    assert await service._verify_prescription_materials([project_search_material(stale)]) == []
    row["status"] = False
    assert await service._verify_prescription_materials([project_search_material(row)]) == []
