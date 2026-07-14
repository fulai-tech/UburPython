"""音频业务编排层（AudioService）。

创建/更新写路径：HTTP → Mongo somni_audio_materials → EsSync（有 audio_url 时）
删除：仍走 comm gRPC + ES delete
读路径：HTTP → 检索缓存 → RetrievalService → ES
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.bionode_grpc_clients import CommClient
from app.cache.audio_search_cache import AudioSearchCache
from app.core.exceptions import MongoNotConfiguredError
from app.es.sync import EsSync
from app.mongo.materials import MaterialsStore
from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioData,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.services.retrieval import RetrievalService


class AudioService:
    """编排 CUD + Search。"""

    def __init__(
        self,
        comm: CommClient,
        es_sync: EsSync,
        retrieval: RetrievalService,
        materials: MaterialsStore | None = None,
        search_cache: AudioSearchCache | None = None,
    ) -> None:
        self._comm = comm
        self._es_sync = es_sync
        self._retrieval = retrieval
        self._materials = materials
        self._search_cache = search_cache

    async def create_audio(self, request: CreateAudioRequest) -> dict[str, Any]:
        store = self._require_materials()
        saved = await store.insert_material(request.to_mongo_doc())
        await self._es_sync.upsert_somni_material(saved["id"], saved)
        await self._clear_search_cache()
        logger.info("已创建音频原料，id={}", saved["id"])
        return saved

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        store = self._require_materials()
        fields = request.to_update_fields()
        saved = await store.update_material(material_id, fields)
        await self._es_sync.upsert_somni_material(material_id, saved)
        await self._clear_search_cache()

    async def delete_audio(self, material_id: str) -> None:
        await self._comm.delete_audio_material(material_id)
        await self._es_sync.delete_audio(material_id)
        await self._clear_search_cache()

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioData:
        cached = await self._get_cached_materials(request)
        if cached is not None:
            return SearchAudioData(materials=cached)
        results = await self._retrieval.search(request)
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

    def _require_materials(self) -> MaterialsStore:
        if self._materials is None:
            raise MongoNotConfiguredError()
        return self._materials
