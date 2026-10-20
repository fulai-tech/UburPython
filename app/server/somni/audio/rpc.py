"""量产音频 gRPC 适配。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct, Value

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.server.somni.audio.catalog import prescription_material_scope
from app.server.somni.audio.hot import parse_hot_kind
from app.server.somni.metadata import parse_language
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.audio.catalog import AudioCatalogService


class AudioRpc(uburnode_somni_pb2_grpc.AudioServiceServicer):
    def __init__(self, service: AudioCatalogService | None) -> None:
        self._service = service
        self._prescription_slots = asyncio.Semaphore(8)

    async def GetAudio(self, request, context):
        service = await self._require(context)
        try:
            language = parse_language(request.language if request.HasField("language") else None)
        except ValueError as exc:
            await abort_invalid(context, str(exc))

        async def _do():
            payload = await service.get_audio(
                **_get_audio_kwargs(request),
                language=language,
            )
            return _to_audio_res(payload)

        return await run_rpc_call(context, _do)

    async def GetAudioTag(self, request, context):
        del request
        service = await self._require(context)

        async def _do():
            payload = await service.get_audio_tag()
            return _to_tag_res(payload)

        return await run_rpc_call(context, _do)

    async def SearchPrescriptionAudio(self, request, context):
        service = await self._require(context)
        if not request.tasks or len(request.tasks) > 64:
            await abort_invalid(context, "tasks 须为 1..64 项")
        ids = [item.task_id.strip() for item in request.tasks]
        if any(not task_id for task_id in ids) or len(set(ids)) != len(ids):
            await abort_invalid(context, "task_id 必须非空且唯一")

        # Reject an invalid tail before starting work for any earlier task.
        for task in request.tasks:
            if (
                task.top_k < 1
                or task.top_k > 20
                or len(task.query_text) > 1000
                or not task.sleep_stage_tags
            ):
                await abort_invalid(context, "invalid prescription search task")

        async def _do():
            batch_slots = asyncio.Semaphore(4)

            async def search(task):
                async with batch_slots, self._prescription_slots:
                    return await service.search_prescription_audio(
                        {
                            "query_text": task.query_text,
                            "sleep_stage_tags": list(task.sleep_stage_tags),
                            "content_tag_codes": list(task.content_tag_codes),
                            "content_tags": list(task.content_tags),
                            "disliked_tags": list(task.disliked_tags),
                            "top_k": task.top_k,
                        }
                    )

            with prescription_material_scope():
                pending = [asyncio.create_task(search(task)) for task in request.tasks]
                try:
                    # gather preserves input order even when later searches finish first.
                    batches = await asyncio.gather(*pending)
                finally:
                    # Includes caller deadline/cancellation and dependency failures.
                    for child in pending:
                        if not child.done():
                            child.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                response = uburnode_somni_pb2.SearchPrescriptionAudioRes()
                for task, candidates in zip(request.tasks, batches, strict=True):
                    result = response.results.add(task_id=task.task_id)
                    for rank, candidate in enumerate(candidates, start=1):
                        struct = ParseDict({**candidate, "retrieval_rank": rank}, Struct())
                        result.candidates.add().CopyFrom(struct)
            return response

        return await run_rpc_call(context, _do)

    async def GetHot(self, request, context):
        service = await self._require(context)
        try:
            language = parse_language(request.language if request.HasField("language") else None)
            kind = parse_hot_kind(request.kind if request.HasField("kind") else None)
        except ValueError as exc:
            await abort_invalid(context, str(exc))

        async def _do():
            payload = await service.get_hot(language=language, kind=kind)
            return _to_hot_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> AudioCatalogService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _get_audio_kwargs(request) -> dict[str, Any]:
    return {
        "page": request.page if request.HasField("page") else None,
        "page_size": request.page_size if request.HasField("page_size") else None,
        "fetch_all": bool(request.fetch_all) if request.HasField("fetch_all") else False,
        "query_text": request.query_text if request.HasField("query_text") else "",
        "tag_code": request.tag_code if request.HasField("tag_code") else "",
        "content_form_codes": [code for code in request.content_form_codes if code],
        "preference_text": request.preference_text if request.HasField("preference_text") else "",
        "dislike_text": request.dislike_text if request.HasField("dislike_text") else "",
    }


def _to_tag_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetAudioTagRes:
    res = uburnode_somni_pb2.GetAudioTagRes()
    for item in payload.get("tags") or []:
        res.tags.append(
            uburnode_somni_pb2.TagDictItem(
                type=str(item.get("type") or ""),
                code=str(item.get("code") or ""),
                name=str(item.get("name") or ""),
                name_en=str(item.get("name_en") or ""),
                id=str(item.get("id") or ""),
                parent_tag_id=_to_value(item.get("parent_tag_id")),
                status=str(item.get("status") or ""),
            )
        )
    return res


def _to_value(value: Any) -> Value:
    result = Value()
    if value is None:
        result.null_value = 0
    else:
        result.string_value = str(value)
    return result


def _to_hot_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetHotRes:
    res = uburnode_somni_pb2.GetHotRes()
    for item in payload.get("items") or []:
        res.items.append(
            uburnode_somni_pb2.HotKeyword(
                keyword=str(item.get("keyword") or ""),
                score=int(item.get("score") or 0),
            )
        )
    return res


def _to_audio_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetAudioRes:
    res = uburnode_somni_pb2.GetAudioRes(
        page=int(payload.get("page") or 1),
        page_size=int(payload.get("page_size") or 0),
        total=int(payload.get("total") or 0),
    )
    for item in payload.get("list") or []:
        entry = uburnode_somni_pb2.AudioListItem(
            id=str(item.get("id") or ""),
            audio_name=str(item.get("audio_name") or ""),
            audio_url=str(item.get("audio_url") or ""),
            cover_url=str(item.get("cover_url") or ""),
            description=str(item.get("description") or ""),
            vip=int(item.get("vip") or 0),
        )
        for tag in item.get("tag") or []:
            entry.tag.append(str(tag))
        res.list.append(entry)
    return res
