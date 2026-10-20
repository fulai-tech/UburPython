"""量产 AudioRpc。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.server.somni.audio.rpc import AudioRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    return MagicMock()


def _make_rpc() -> tuple[AudioRpc, MagicMock]:
    service = MagicMock()
    service.get_audio = AsyncMock()
    service.get_audio_tag = AsyncMock()
    service.get_hot = AsyncMock()
    return AudioRpc(service), service


@pytest.mark.asyncio
async def test_get_audio_passes_language_en() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(return_value={"list": [], "page": 1, "page_size": 20, "total": 0})
    req = uburnode_somni_pb2.GetAudioReq(page=1, page_size=20, language="en")
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=1,
        page_size=20,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=[],
        preference_text="",
        dislike_text="",
        language="en",
    )


@pytest.mark.asyncio
async def test_get_audio_rejects_invalid_language() -> None:
    rpc, service = _make_rpc()
    ctx = _context()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    req = uburnode_somni_pb2.GetAudioReq(language="fr")
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetAudio(req, ctx)
    service.get_audio.assert_not_called()
    ctx.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_audio_passes_page_and_query() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [
                {
                    "id": "m1",
                    "audio_name": "雨声",
                    "audio_url": "https://cdn.example/a.mp3",
                    "cover_url": "https://cdn.example/a.png",
                    "description": "短描述",
                    "vip": 0,
                    "tag": ["自然声", "中雨/稳定雨声"],
                }
            ],
            "page": 1,
            "page_size": 20,
            "total": 1,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(page=1, page_size=20, query_text="雨声")
    res = await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=1,
        page_size=20,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
        content_form_codes=[],
        preference_text="",
        dislike_text="",
        language="zh",
    )
    assert res.list[0].audio_name == "雨声"
    assert res.list[0].vip == 0
    assert res.list[0].description == "短描述"
    assert res.total == 1
    assert res.page == 1
    assert res.page_size == 20
    assert list(res.list[0].tag) == ["自然声", "中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_fetch_all() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [],
            "page": 1,
            "page_size": 0,
            "total": 0,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(fetch_all=True)
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=None,
        page_size=None,
        fetch_all=True,
        query_text="",
        tag_code="",
        content_form_codes=[],
        preference_text="",
        dislike_text="",
        language="zh",
    )


@pytest.mark.asyncio
async def test_get_audio_passes_tag_code() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [],
            "page": 1,
            "page_size": 20,
            "total": 0,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(tag_code="steady_rain")
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=None,
        page_size=None,
        fetch_all=False,
        query_text="",
        tag_code="steady_rain",
        content_form_codes=[],
        preference_text="",
        dislike_text="",
        language="zh",
    )


@pytest.mark.asyncio
async def test_get_audio_tag_maps_fields() -> None:
    rpc, service = _make_rpc()
    service.get_audio_tag = AsyncMock(
        return_value={
            "tags": [
                {
                    "type": "content_form",
                    "code": "natural_sound",
                    "name": "自然声",
                    "name_en": "Natural Sound",
                    "id": "root-natural",
                    "parent_tag_id": None,
                    "status": "启用",
                }
            ]
        }
    )
    res = await rpc.GetAudioTag(uburnode_somni_pb2.GetAudioTagReq(), _context())
    service.get_audio_tag.assert_awaited_once_with()
    assert res.tags[0].code == "natural_sound"
    assert res.tags[0].name_en == "Natural Sound"
    assert res.tags[0].id == "root-natural"
    assert res.tags[0].parent_tag_id.WhichOneof("kind") == "null_value"
    assert res.tags[0].status == "启用"


@pytest.mark.asyncio
async def test_get_hot_passes_language_en() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(return_value={"items": []})
    await rpc.GetHot(uburnode_somni_pb2.GetHotReq(language="en"), _context())
    service.get_hot.assert_awaited_once_with(language="en", kind="query")


@pytest.mark.asyncio
async def test_get_hot_passes_kind_tag() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(return_value={"items": []})
    await rpc.GetHot(
        uburnode_somni_pb2.GetHotReq(language="zh", kind="tag"),
        _context(),
    )
    service.get_hot.assert_awaited_once_with(language="zh", kind="tag")


@pytest.mark.asyncio
async def test_get_hot_maps_items() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(return_value={"items": [{"keyword": "雨声", "score": 5}]})
    res = await rpc.GetHot(uburnode_somni_pb2.GetHotReq(), _context())
    service.get_hot.assert_awaited_once_with(language="zh", kind="query")
    assert res.items[0].keyword == "雨声"
    assert res.items[0].score == 5


@pytest.mark.asyncio
async def test_get_hot_rejects_invalid_kind() -> None:
    rpc, service = _make_rpc()
    ctx = _context()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetHot(uburnode_somni_pb2.GetHotReq(kind="other"), ctx)
    service.get_hot.assert_not_called()
    ctx.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_prescription_audio_batches_by_task_without_hot_api() -> None:
    rpc, service = _make_rpc()
    service.search_prescription_audio = AsyncMock(
        return_value=[
            {
                "id": "audio-1",
                "audio_name": "rain",
                "audio_url": "https://cdn.example/rain.mp3",
                "tags": ["steady_rain"],
                "_score": 0.87,
            }
        ]
    )
    request = uburnode_somni_pb2.SearchPrescriptionAudioReq(
        tasks=[
            uburnode_somni_pb2.PrescriptionAudioSearchTask(
                task_id="task-1",
                problem_code="C1.1",
                intervention_code="breathing_guidance",
                query_text="gentle breathing",
                content_tag_codes=["breathing_guidance"],
                sleep_stage_tags=["unwind"],
                top_k=5,
            )
        ]
    )
    result = await rpc.SearchPrescriptionAudio(request, _context())
    assert len(result.results) == 1
    assert result.results[0].task_id == "task-1"
    assert result.results[0].candidates[0].fields["id"].string_value == "audio-1"
    service.search_prescription_audio.assert_awaited_once()
    service.get_hot.assert_not_called()


def _prescription_request(count=12):
    request = uburnode_somni_pb2.SearchPrescriptionAudioReq()
    for index in range(count):
        request.tasks.add(
            task_id=str(index), query_text=str(index), sleep_stage_tags=["unwind"], top_k=1
        )
    return request


@pytest.mark.asyncio
async def test_prescription_batch_parallelism_is_bounded_and_results_keep_order():
    rpc, service = _make_rpc()
    release = asyncio.Event()
    saturated = asyncio.Event()
    active = peak = 0

    async def search(payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 4:
            saturated.set()
        try:
            await release.wait()
            if payload["query_text"] == "0":
                await asyncio.sleep(0)
            return [{"id": payload["query_text"]}]
        finally:
            active -= 1

    service.search_prescription_audio = search
    task = asyncio.create_task(rpc.SearchPrescriptionAudio(_prescription_request(), _context()))
    try:
        await asyncio.wait_for(saturated.wait(), 1)
        release.set()
        result = await task
        assert peak == 4
        assert [item.task_id for item in result.results] == [str(i) for i in range(12)]
        assert [item.candidates[0]["id"] for item in result.results] == [str(i) for i in range(12)]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_prescription_batch_validates_all_tasks_before_retrieval():
    rpc, service = _make_rpc()
    service.search_prescription_audio = AsyncMock(return_value=[])
    request = _prescription_request(2)
    request.tasks[1].top_k = 21
    context = _context()
    context.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    with pytest.raises(grpc.aio.AbortError):
        await rpc.SearchPrescriptionAudio(request, context)
    service.search_prescription_audio.assert_not_awaited()
    assert context.abort.call_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_prescription_cancellation_or_failure_drains_started_children(failure):
    rpc, service = _make_rpc()
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def search(payload):
        nonlocal active
        active += 1
        if active == 4:
            started.set()
        try:
            await started.wait()
            if failure and payload["query_text"] == "0":
                raise RuntimeError("dependency failed")
            await release.wait()
            return []
        finally:
            active -= 1

    service.search_prescription_audio = search
    context = _context()
    context.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    task = asyncio.create_task(rpc.SearchPrescriptionAudio(_prescription_request(), context))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if not failure:
            task.cancel()
        with pytest.raises((asyncio.CancelledError, grpc.aio.AbortError)):
            await task
        assert active == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_prescription_parallel_batches_share_service_limit():
    rpc, service = _make_rpc()
    saturated = asyncio.Event()
    release = asyncio.Event()
    active = peak = 0

    async def search(payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 8:
            saturated.set()
        try:
            await release.wait()
            return []
        finally:
            active -= 1

    service.search_prescription_audio = search
    pending = [
        asyncio.create_task(rpc.SearchPrescriptionAudio(_prescription_request(), _context()))
        for _ in range(3)
    ]
    try:
        await asyncio.wait_for(saturated.wait(), 1)
        release.set()
        await asyncio.gather(*pending)
        assert peak == 8
        assert active == 0
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_grpc_deadline_cancels_and_drains_parallel_searches():
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2_grpc

    rpc, service = _make_rpc()
    started = asyncio.Event()
    drained = asyncio.Event()
    active = 0

    async def search(payload):
        nonlocal active
        active += 1
        if active == 4:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            if active == 0:
                drained.set()

    service.search_prescription_audio = search
    server = grpc.aio.server()
    uburnode_somni_pb2_grpc.add_AudioServiceServicer_to_server(rpc, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = uburnode_somni_pb2_grpc.AudioServiceStub(channel)
            call = stub.SearchPrescriptionAudio(_prescription_request(), timeout=0.3)
            await asyncio.wait_for(started.wait(), 1)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            await asyncio.wait_for(drained.wait(), 1)
            assert active == 0
    finally:
        await server.stop(0)
