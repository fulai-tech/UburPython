"""音频业务编排层（AudioService）。

写路径：HTTP → comm gRPC → Mongo → EsSync → ES
读路径：HTTP → RetrievalService → ES（不经过 comm，规范 §一）
"""

from __future__ import annotations

from loguru import logger

from app.comm.client import CommClient
from app.es.sync import EsSync
from app.schemas.audio import (
    CreateAudioRequest,
    CreateAudioResponse,
    EvidenceLevel,
    EVIDENCE_WEIGHT_MAP,
    SearchAudioRequest,
    SearchAudioResponse,
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
    ) -> None:
        self._comm = comm
        self._es_sync = es_sync
        self._retrieval = retrieval

    async def create_audio(self, request: CreateAudioRequest) -> CreateAudioResponse:
        weight = request.recommend_weight
        if weight is None:
            weight = EVIDENCE_WEIGHT_MAP[request.evidence_level]

        await self._comm.create_audio_material(
            category_code=request.category_code,
            noise_color=request.noise_color,
            name=request.audio_name,
            description=request.description,
            tags=request.tags,
            audio_url=request.audio_url,
        )

        material_id = await self._resolve_latest_material_id(request.audio_name)
        logger.info("已创建音频原料，id={}", material_id)

        await self._es_sync.upsert_audio(
            material_id,
            audio_url=request.audio_url,
            audio_name=request.audio_name,
            flat_tags=request.tags,
            evidence_level=request.evidence_level.value,
            recommend_weight=weight,
        )
        return CreateAudioResponse(id=material_id)

    async def update_audio(self, material_id: str, request: UpdateAudioRequest) -> None:
        await self._comm.update_audio_material(
            material_id,
            category_code=request.category_code,
            noise_color=request.noise_color,
            name=request.audio_name,
            description=request.description,
            tags=request.tags,
            audio_url=request.audio_url,
            status=request.status,
        )

        # ES 需全量文档：从 comm 读回合并后的真值再 upsert
        material = await self._comm.get_audio_material(material_id)
        evidence = request.evidence_level.value if request.evidence_level else "C"
        weight = request.recommend_weight
        if weight is None and request.evidence_level:
            weight = EVIDENCE_WEIGHT_MAP.get(
                request.evidence_level,
                EVIDENCE_WEIGHT_MAP[EvidenceLevel.C],
            )
        if weight is None:
            weight = 0.45

        audio_url = request.audio_url or material.audio_info.meta_data.url
        audio_name = request.audio_name or material.name
        tags = request.tags if request.tags is not None else list(material.tags)

        await self._es_sync.upsert_audio(
            material_id,
            audio_url=audio_url,
            audio_name=audio_name,
            flat_tags=tags,
            evidence_level=evidence,
            recommend_weight=weight,
        )

    async def delete_audio(self, material_id: str) -> None:
        await self._comm.delete_audio_material(material_id)
        await self._es_sync.delete_audio(material_id)

    async def search_audio(self, request: SearchAudioRequest) -> SearchAudioResponse:
        results = await self._retrieval.search(request)
        return SearchAudioResponse(results=results)

    async def _resolve_latest_material_id(self, name: str) -> str:
        """comm CreateAudioMaterial 返回 EmptyRes，按名称反查 id（临时方案）。"""
        materials = await self._comm.list_audio_materials_by_name(name)
        if not materials:
            raise RuntimeError(f"创建成功但按名称未找到原料：{name}")
        return materials[0].id
