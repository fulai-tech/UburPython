"""Hybrid prescription recall preserves hard constraints, score meaning and language basis."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.es.search import SEARCH_CANDIDATE_SOURCE_INCLUDES, EsSearch
from app.schemas.audio import SearchAudioRequest
from app.services.hybrid_recall import merge_documents
from app.services.retrieval import RetrievalService, _vector_key


def material(identifier, *, code="sleep_music", score=None):
    row = {
        "_id": identifier,
        "audio_name": identifier,
        "description": "完整描述",
        "audio_name_en": "English " + identifier,
        "short_description_en": "English short description",
        "audio_url": "https://example.test/" + identifier + ".mp3",
        "status": True,
        "sleep_stage_tags": [{"code": "unwind", "tag_id": "phase-unwind"}],
        "content_form_tags": [{"code": code, "tag_id": code}],
    }
    if score is not None:
        row["_description_score"] = score
    return row


def service(bm25=None, vector=None):
    es = MagicMock(spec=EsSearch)
    es.parse_tags.side_effect = EsSearch.parse_tags
    es.list_content_tag_vectors = AsyncMock(return_value=[])
    es.search_get_audio_bm25 = AsyncMock(return_value=bm25 or [])
    es.search_by_description_vector = AsyncMock(return_value=vector or [])
    encoder = MagicMock()
    encoder.encode = AsyncMock(side_effect=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    encoder.encode_one = AsyncMock(
        side_effect=lambda text, *, language: [1.0, 0.0] if language == "en" else [1.0, 0.0, 0.0]
    )
    result = RetrievalService(
        es,
        encoder,
        Settings(search_sleep_stage_filter_enabled=True, embedding_dim=3, embedding_dim_en=2),
    )
    result._fetch_step1_candidates = AsyncMock(return_value=[])
    return result, es, encoder


def request(text="relaxing instrumental music", **kwargs):
    return SearchAudioRequest(query_text=text, sleep_stage_tags=["unwind"], top_k=10, **kwargs)


@pytest.mark.asyncio
async def test_keyword_only_and_vector_only_preserve_original_scores():
    svc, es, encoder = service([material("keyword")], [material("semantic", score=0.8)])
    output = await svc.search_prescription(request())
    assert [row["_id"] for row in output] == ["semantic", "keyword"]
    assert output[0]["audio_name_en"] == "English semantic"
    assert output[0]["short_description_en"] == "English short description"
    bm25 = es.search_get_audio_bm25.await_args.kwargs
    vector = es.search_by_description_vector.await_args.kwargs
    assert bm25["sleep_stage_tags"] == vector["sleep_stage_tags"] == ["unwind"]
    assert bm25["source_includes"] == SEARCH_CANDIDATE_SOURCE_INCLUDES
    assert vector["language"] == bm25["language"] == "en"
    assert len(es.search_by_description_vector.await_args.args[0]) == 2
    assert encoder.encode.await_args.args[0] == ["relaxing instrumental music"]
    assert all("_rrf_score" not in row for row in output)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", ["bm25", "vector"])
async def test_one_failed_lane_keeps_legal_material(failed):
    svc, es, _ = service([material("keyword")], [material("semantic", score=0.9)])
    getattr(
        es, "search_get_audio_bm25" if failed == "bm25" else "search_by_description_vector"
    ).side_effect = RuntimeError("synthetic")
    assert len(await svc.search_prescription(request())) == 1


@pytest.mark.asyncio
async def test_failed_english_encoder_does_not_block_bm25():
    svc, es, encoder = service([material("keyword")])
    encoder.encode_one.side_effect = FileNotFoundError("English model unavailable")
    assert [m["_id"] for m in await svc.search_prescription(request())] == ["keyword"]
    es.search_by_description_vector.assert_not_awaited()


@pytest.mark.asyncio
async def test_both_lanes_failed_is_distinct_from_empty_results():
    svc, es, _ = service()
    assert await svc.search_prescription(request()) == []
    es.search_get_audio_bm25.side_effect = RuntimeError("bm25")
    es.search_by_description_vector.side_effect = RuntimeError("vector")
    with pytest.raises(AppError) as error:
        await svc.search_prescription(request())
    assert error.value.status_code == 503


def test_rrf_deduplicates_without_turning_bm25_score_into_similarity():
    rows = merge_documents([material("keyword"), material("both")], [material("both", score=0.7)])
    assert [r["_id"] for r in rows] == ["both", "keyword"]
    assert rows[0]["_description_score"] == 0.7
    assert "_description_score" not in rows[1]


@pytest.mark.asyncio
async def test_pure_tags_and_only_negative_text_skip_hybrid_lanes():
    svc, es, _ = service()
    svc._search_tag_only = AsyncMock(return_value=[])
    await svc.search_prescription(request(""))
    await svc.search_prescription(request("不要雨声"))
    es.search_get_audio_bm25.assert_not_awaited()
    es.search_by_description_vector.assert_not_awaited()
    dislikes = set(svc._search_tag_only.await_args.args[0].disliked_tags)
    assert "steady_light_rain" in dislikes
    assert "gentle_ocean_wave" not in dislikes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,codes,expected",
    [
        ("白噪音", ["white_noise"], {"white", "rain"}),
        ("白噪音和粉噪音", ["white_noise", "pink_noise"], {"white", "pink", "rain"}),
        (
            "白噪音和粉噪音和雨声",
            ["white_noise", "pink_noise", "steady_light_rain"],
            {"white", "pink", "rain"},
        ),
    ],
)
async def test_pure_and_mixed_color_requests_do_not_expand_prohibitions(text, codes, expected):
    rows = [
        material("white", code="white_noise"),
        material("pink", code="pink_noise"),
        material("rain", code="steady_light_rain"),
    ]
    svc, _, _ = service(rows)
    result = await svc.search_prescription(request(text, content_tags=codes))
    assert {r["_id"] for r in result} == expected


def test_language_model_and_policy_are_in_natural_cache_identity():
    assert (
        len(
            {
                _vector_key("same", natural=True, language=language, model=m)
                for language in ("zh", "en")
                for m in ("old", "new")
            }
        )
        == 4
    )


@pytest.mark.asyncio
async def test_bm25_es_query_filters_before_truncation_and_returns_full_english_metadata():
    client = MagicMock()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    es = EsSearch(client, Settings())
    await es.search_get_audio_bm25(
        "soft rain",
        language="en",
        size=10,
        sleep_stage_tags=["unwind"],
        source_includes=SEARCH_CANDIDATE_SOURCE_INCLUDES,
    )
    body = client.search.await_args.kwargs["body"]
    assert {"term": {"status": True}} in body["query"]["bool"]["filter"]
    assert "sleep_stage" in str(body["query"]["bool"]["filter"])
    assert "description_text_en" in str(body["query"]["bool"]["must"])
    assert {"audio_name_en", "short_description_en", "description", "sleep_stage_tags"} <= set(
        body["_source"]["includes"]
    )


@pytest.mark.asyncio
async def test_wrong_dimension_is_not_sent_to_es():
    svc, es, encoder = service([material("keyword")])
    encoder.encode_one.return_value = [0.1] * 3
    encoder.encode_one.side_effect = None
    assert (await svc.search_prescription(request()))[0]["_id"] == "keyword"
    es.search_by_description_vector.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_audio_rpc_invokes_hybrid_and_preserves_english_fields():
    import grpc

    from app.server.somni.audio.catalog import AudioCatalogService
    from app.server.somni.audio.rpc import AudioRpc
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2_grpc as rpc

    svc, es, _ = service([material("keyword")])
    server = grpc.aio.server()
    rpc.add_AudioServiceServicer_to_server(
        AudioRpc(AudioCatalogService(None, Settings(), retrieval=svc)), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            result = await rpc.AudioServiceStub(channel).SearchPrescriptionAudio(
                pb.SearchPrescriptionAudioReq(
                    tasks=[
                        pb.PrescriptionAudioSearchTask(
                            task_id="synthetic-task",
                            query_text="instrumental",
                            sleep_stage_tags=["unwind"],
                            top_k=1,
                        )
                    ]
                )
            )
        row = result.results[0].candidates[0]
        assert result.results[0].task_id == "synthetic-task"
        assert row["audio_name_en"] == "English keyword"
        assert row["short_description_en"] == "English short description"
        es.search_get_audio_bm25.assert_awaited_once()
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_prescription_language_field_is_not_overridden_by_catalogue_setting():
    client = MagicMock()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    es = EsSearch(client, Settings(get_audio_description_vector_field="description_vector"))
    await es.search_by_description_vector(
        [0.1, 0.2], sleep_stage_tags=["unwind"], size=5, language="en", allow_field_override=False
    )
    body = client.search.await_args.kwargs["body"]
    assert body["knn"]["field"] == "description_vector_en"
    assert "sleep_stage" in str(body["knn"]["filter"])
