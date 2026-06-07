"""音频业务编排层（AudioService）。

写路径：HTTP → comm gRPC → Mongo → EsSync → ES
读路径：HTTP → RetrievalService → ES（不经过 comm，规范 §一）
"""

from __future__ import annotations

from loguru import logger

from app.comm.client import CommClient
from app.core.exceptions import CommMaterialNotFoundError
from app.es.sync import EsSync
from app.schemas.audio import (
    AudioMaterialData,
    CreateAudioRequest,
    SearchAudioData,
    SearchAudioRequest,
    UpdateAudioRequest,
    WriteAudioRequest,
)
from app.services.retrieval import RetrievalService


class AudioService:
    """编排 CUD + Search。"""

    def __init__(
        self,
        comm: CommClient,
        es_sync: EsSync,
        retrieval: RetrievalService,
    ) -> None:
        self._comm = comm
        self._es_sync = es_sync
        self._retrieval = retrieval

    async def create_audio(self, request: CreateAudioRequest) -> AudioMaterialData:
        await self._comm.create_audio_material(
            category_code=request.category_code,
            noise_color=request.resolved_noise_color(),
            name=request.name,
            description=request.description,
            tags=request.flat_tags(),
            audio_info=request.audio_info,
        )

        material_id = await self._resolve_latest_material_id(request.name)
        logger.info("已创建音频原料，id={}", material_id)

        await self._sync_es(material_id, request)
        material = await self._comm.get_audio_material(material_id)
        return AudioMaterialData.from_comm_material(material)

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        await self._comm.update_audio_material(
            material_id,
            category_code=request.category_code,
            noise_color=request.resolved_noise_color(),
            name=request.name,
            description=request.description,
            tags=request.flat_tags(),
            audio_info=request.audio_info,
        )
        await self._sync_es(material_id, request)

    async def delete_audio(self, material_id: str) -> None:
        await self._comm.delete_audio_material(material_id)
        await self._es_sync.delete_audio(material_id)

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioData:
        results = await self._retrieval.search(request)
        return SearchAudioData(audios=results)

    async def _sync_es(self, material_id: str, request: WriteAudioRequest) -> None:
        await self._es_sync.upsert_audio(
            material_id,
            audio_url=request.audio_url,
            audio_name=request.name,
            flat_tags=request.flat_tags(),
            evidence_level=request.evidence_level.value,
            recommend_weight=request.resolved_recommend_weight(),
        )

    async def _resolve_latest_material_id(self, name: str) -> str:
        """comm CreateAudioMaterial 返回 EmptyRes，按名称反查 id（临时方案）。"""
        materials = await self._comm.list_audio_materials_by_name(name)
        if not materials:
            raise CommMaterialNotFoundError(name)
        return materials[0].id
