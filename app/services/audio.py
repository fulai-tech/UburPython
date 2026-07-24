"""音频业务编排层（AudioService）。

写路径：HTTP → comm gRPC（Create/Update/Delete）→ EsSync
读路径：HTTP → 检索缓存 → RetrievalService → ES
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.bionode_grpc_clients import CommClient
from app.cache.audio_search_cache import AudioSearchCache
from app.cache.sleep_stage_cache import SleepStageCandidateCache
from app.cache.sleep_stage_refresh import DebouncedSleepStageCacheRefresh
from app.core.config import Settings
from app.core.exceptions import CommMaterialNotFoundError
from app.es.sync import EsSync
from app.mongo.materials import MaterialsStore
from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioData,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.services.retrieval import RetrievalService

# 与 Mongo 写入默认对齐，保证创建 HTTP 响应字段形状不变
_CREATE_RESPONSE_DEFAULTS: dict[str, Any] = {
    "status": True,
    "audio_url": "",
    "operation_type": 0,
    "sleep_stage_tags": [],
    "content_form_tags": [],
    "mechanism_tags": [],
    "audio_engineering_tags": [],
    "medical_risk_tags": [],
    "evidence_level_tags": [],
}


class AudioService:
    """编排 CUD + Search。"""

    def __init__(
        self,
        comm: CommClient,
        es_sync: EsSync,
        retrieval: RetrievalService,
        materials: MaterialsStore | None = None,
        search_cache: AudioSearchCache | None = None,
        sleep_stage_cache: SleepStageCandidateCache | None = None,
        *,
        sleep_stage_rewarm_delay_sec: float | None = None,
    ) -> None:
        self._comm = comm
        self._es_sync = es_sync
        self._retrieval = retrieval
        self._materials = materials
        self._search_cache = search_cache
        self._sleep_stage_cache = sleep_stage_cache
        delay = (
            sleep_stage_rewarm_delay_sec
            if sleep_stage_rewarm_delay_sec is not None
            else Settings().sleep_stage_cache_rewarm_delay_sec
        )
        self._sleep_stage_refresh = DebouncedSleepStageCacheRefresh(
            clear=self._retrieval.clear_sleep_stage_cache,
            warm=self._retrieval.warm_sleep_stage_cache,
            delay_sec=delay,
        )

    async def create_audio(self, request: CreateAudioRequest) -> dict[str, Any]:
        await self._comm.create_audio_material(request)
        material_id = await self._resolve_created_id(request.audio_name)
        saved = _create_response_doc(material_id, request)
        await self._es_sync.upsert_somni_material(material_id, saved)
        await self._invalidate_candidate_caches()
        logger.info("已创建音频原料，id={}", material_id)
        return saved

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        await self._comm.update_audio_material(material_id, request)
        saved = {"id": material_id, **request.model_dump(exclude_unset=True)}
        await self._es_sync.upsert_somni_material(material_id, saved)
        await self._invalidate_candidate_caches()

    async def delete_audio(self, material_id: str) -> None:
        await self._comm.delete_audio_material(material_id)
        await self._es_sync.delete_audio(material_id)
        await self._invalidate_candidate_caches()

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioData:
        cached = await self._get_cached_materials(request)
        if cached is not None:
            return SearchAudioData(materials=cached)
        results = await self._retrieval.search(request)
        # 空结果不写入缓存，避免长时间缓存「无命中」导致误伤
        if results:
            await self._set_cached_materials(request, results)
        return SearchAudioData(materials=results)

    async def _resolve_created_id(self, audio_name: str) -> str:
        materials = await self._comm.list_audio_materials_by_name(audio_name)
        if not materials:
            raise CommMaterialNotFoundError(audio_name)
        return materials[0].id

    async def _get_cached_materials(
        self, request: SearchAudioRequest
    ) -> list[dict[str, Any]] | None:
        if self._search_cache is None:
            return None
        try:
            return await self._search_cache.get(request)
        except Exception as exc:
            logger.warning("检索缓存读取失败，回退实时检索：{}", exc)
            return None

    async def _set_cached_materials(
        self, request: SearchAudioRequest, materials: list[dict[str, Any]]
    ) -> None:
        if self._search_cache is None:
            return
        try:
            await self._search_cache.set(request, materials)
        except Exception as exc:
            logger.warning("检索缓存写入失败：{}", exc)

    async def _clear_search_cache(self) -> None:
        if self._search_cache is None:
            return
        try:
            await self._search_cache.clear_all()
        except Exception as exc:
            logger.error("清除检索缓存失败：{}", exc)

    async def _invalidate_candidate_caches(self) -> None:
        """CUD 后立即清缓存；睡眠阶段候选延时去抖重建，避免频繁写入反复打 ES。"""
        await self._clear_search_cache()
        await self._sleep_stage_refresh.invalidate()


def _create_response_doc(material_id: str, request: CreateAudioRequest) -> dict[str, Any]:
    payload = {**_CREATE_RESPONSE_DEFAULTS, **request.to_mongo_doc()}
    payload["id"] = material_id
    return payload
