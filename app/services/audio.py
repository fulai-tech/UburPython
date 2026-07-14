"""音频业务编排层（AudioService）。

创建/更新写路径：HTTP → Mongo somni_audio_materials → EsSync（有 audio_url 时）
删除：仍走 comm gRPC + ES delete
读路径：HTTP → RetrievalService → ES
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.bionode_grpc_clients import CommClient
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
    ) -> None:
        self._comm = comm
        self._es_sync = es_sync
        self._retrieval = retrieval
        self._materials = materials

    async def create_audio(self, request: CreateAudioRequest) -> dict[str, Any]:
        store = self._require_materials()
        saved = await store.insert_material(request.to_mongo_doc())
        await self._es_sync.upsert_somni_material(saved["id"], saved)
        logger.info("已创建音频原料，id={}", saved["id"])
        return saved

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        store = self._require_materials()
        fields = request.to_update_fields()
        saved = await store.update_material(material_id, fields)
        await self._es_sync.upsert_somni_material(material_id, saved)

    async def delete_audio(self, material_id: str) -> None:
        await self._comm.delete_audio_material(material_id)
        await self._es_sync.delete_audio(material_id)

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioData:
        results = await self._retrieval.search(request)
        return SearchAudioData(materials=results)

    def _require_materials(self) -> MaterialsStore:
        if self._materials is None:
            raise MongoNotConfiguredError()
        return self._materials
