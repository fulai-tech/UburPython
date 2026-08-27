"""功能手板音频业务：直连 Mongo + 本侧 ES / 缓存。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.cache.audio_search_cache import AudioSearchCache
from app.cache.sleep_stage_cache import SleepStageCandidateCache
from app.cache.sleep_stage_refresh import DebouncedSleepStageCacheRefresh
from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError
from app.es.sync import EsSync
from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioData,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.server.handboard.audio.store import MaterialsStore
from app.services.retrieval import RetrievalService

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
    """编排 CUD + Search（无 BioNode）。"""

    def __init__(
        self,
        materials: MaterialsStore | None,
        es_sync: EsSync,
        retrieval: RetrievalService,
        search_cache: AudioSearchCache | None = None,
        sleep_stage_cache: SleepStageCandidateCache | None = None,
        *,
        sleep_stage_rewarm_delay_sec: float | None = None,
    ) -> None:
        self._materials = materials
        self._es_sync = es_sync
        self._retrieval = retrieval
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

    def _require_store(self) -> MaterialsStore:
        if self._materials is None:
            raise MongoNotConfiguredError()
        return self._materials

    async def create_audio(self, request: CreateAudioRequest) -> dict[str, Any]:
        store = self._require_store()
        saved = await store.insert_material(request.to_mongo_doc())
        await self._es_sync.upsert_somni_material(saved["id"], saved)
        await self._invalidate_candidate_caches()
        logger.info("已创建音频原料，id={}", saved["id"])
        return {**_CREATE_RESPONSE_DEFAULTS, **saved}

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        store = self._require_store()
        saved = await store.update_material(material_id, request.to_update_fields())
        await self._es_sync.upsert_somni_material(material_id, saved)
        await self._invalidate_candidate_caches()

    async def delete_audio(self, material_id: str) -> None:
        store = self._require_store()
        await store.delete_material(material_id)
        await self._es_sync.delete_audio(material_id)
        await self._invalidate_candidate_caches()

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioData:
        cached = await self._get_cached_materials(request)
        if cached is not None:
            return SearchAudioData(materials=cached)
        results = await self._retrieval.search(request)
        if results:
            await self._set_cached_materials(request, results)
        return SearchAudioData(materials=results)

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
        await self._clear_search_cache()
        await self._sleep_stage_refresh.invalidate()
