#!/usr/bin/env python3
"""comm 原料 → ES 差异同步（单文件：适配、备份、对账、向量化、定时调度）。

以源库 _id 为准，只读不写源库：
  - ES 有、源库无 → 删 ES
  - 源库有 → 比差异，有变才 upsert（标签向量同名 label 复用，见 EsSync）
  - 每次同步前删旧备份并写新备份至 data/sync_backup/

服务启动后按 SYNC_INTERVAL_DAYS 注册定时任务；也可手动执行本脚本。

用法:
  .venv/bin/python scripts/sync_es_from_comm.py
  .venv/bin/python scripts/sync_es_from_comm.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from elasticsearch import AsyncElasticsearch  # noqa: E402
from loguru import logger  # noqa: E402

from app.bionode_grpc_clients import CommClient  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.tags import DIMENSION_FIELDS, dimensions_from_flat_tags  # noqa: E402
from app.embedding.encoder import Encoder  # noqa: E402
from app.es.search import EsSearch  # noqa: E402
from app.es.sync import EsSync  # noqa: E402
from app.schemas.audio import EVIDENCE_WEIGHT_MAP, EvidenceLevel  # noqa: E402

if TYPE_CHECKING:
    from app.main import AppState

CONTENT_TAG_PREFIX = "content:"

_scheduler: AsyncIOScheduler | None = None


# ---------------------------------------------------------------------------
# 原料 → ES 载荷
# ---------------------------------------------------------------------------


class AudioMaterialLike(Protocol):
    id: str
    category_code: int
    name: str
    tags: list[str]

    @property
    def audio_info(self) -> object: ...


@dataclass(frozen=True)
class EsSyncPayload:
    doc_id: str
    audio_url: str
    audio_name: str
    flat_tags: list[str]
    evidence_level: str
    recommend_weight: float


@dataclass(frozen=True)
class PayloadBuildResult:
    """原料列表 → ES 写入载荷；区分无效跳过与重复 _id 去重。"""

    payloads: dict[str, EsSyncPayload]
    skipped_invalid: int
    deduped: int

    @property
    def unique_count(self) -> int:
        return len(self.payloads)


@dataclass(frozen=True)
class SyncJobResult:
    total_fetched: int
    unique_sources: int
    skipped_invalid: int
    deduped: int
    deleted: int
    created: int
    updated: int
    unchanged: int
    failed: int


def content_tags_to_flat(content_tags: list[str]) -> list[str]:
    flat: list[str] = []
    for tag in content_tags:
        label = tag.strip()
        if not label:
            continue
        if label.startswith(CONTENT_TAG_PREFIX):
            flat.append(label)
        else:
            flat.append(f"{CONTENT_TAG_PREFIX}{label}")
    return flat


def _extract_audio_url(material: AudioMaterialLike) -> str:
    audio_info = material.audio_info
    meta_data = getattr(audio_info, "meta_data", None)
    if meta_data is None:
        return ""
    return getattr(meta_data, "url", "") or ""


def build_payloads_from_materials(materials: list[object]) -> PayloadBuildResult:
    """将 comm 原料转为 ES 载荷；重复 _id 保留首条。"""
    payloads: dict[str, EsSyncPayload] = {}
    skipped_invalid = 0
    deduped = 0
    for material in materials:
        payload = material_to_es_payload(material)
        if payload is None:
            skipped_invalid += 1
            logger.warning(
                "跳过无效原料，id={}，name={}（缺少 id 或 audio_url）",
                getattr(material, "id", ""),
                getattr(material, "name", ""),
            )
            continue
        if payload.doc_id in payloads:
            deduped += 1
            logger.warning(
                "跳过重复原料，id={}，name={}（comm 返回重复 _id，保留首条）",
                payload.doc_id,
                getattr(material, "name", ""),
            )
            continue
        payloads[payload.doc_id] = payload
    return PayloadBuildResult(
        payloads=payloads,
        skipped_invalid=skipped_invalid,
        deduped=deduped,
    )


def material_to_es_payload(material: AudioMaterialLike) -> EsSyncPayload | None:
    doc_id = (material.id or "").strip()
    audio_url = _extract_audio_url(material).strip()
    if not doc_id or not audio_url:
        return None
    flat_tags = content_tags_to_flat(list(material.tags))
    default_level = EvidenceLevel.C
    return EsSyncPayload(
        doc_id=doc_id,
        audio_url=audio_url,
        audio_name=material.name or "",
        flat_tags=flat_tags,
        evidence_level=default_level.value,
        recommend_weight=EVIDENCE_WEIGHT_MAP[default_level],
    )


# ---------------------------------------------------------------------------
# ES 文档 diff（只比 label，不比 vector_id）
# ---------------------------------------------------------------------------


def build_compare_snapshot(
    *,
    audio_url: str,
    audio_name: str,
    flat_tags: list[str],
    evidence_level: str,
    recommend_weight: float,
) -> dict[str, Any]:
    dimensions = dimensions_from_flat_tags(flat_tags)
    return {
        "audio_url": audio_url,
        "audio_name": audio_name,
        "evidence_level": evidence_level,
        "recommend_weight": recommend_weight,
        "tags": {dim: sorted(dimensions[dim]) for dim in DIMENSION_FIELDS},
    }


def compare_snapshot_from_es_source(source: dict[str, Any]) -> dict[str, Any]:
    tags_raw = source.get("tags") or {}
    tags: dict[str, list[str]] = {}
    for dim in DIMENSION_FIELDS:
        items = tags_raw.get(dim) or []
        labels = [str(item.get("label", "")).strip() for item in items if item.get("label")]
        tags[dim] = sorted(labels)
    return {
        "audio_url": source.get("audio_url", ""),
        "audio_name": source.get("audio_name", ""),
        "evidence_level": source.get("evidence_level", ""),
        "recommend_weight": source.get("recommend_weight"),
        "tags": tags,
    }


def audio_documents_differ(desired: dict[str, Any], existing_source: dict[str, Any]) -> bool:
    return desired != compare_snapshot_from_es_source(existing_source)


# ---------------------------------------------------------------------------
# 本地备份
# ---------------------------------------------------------------------------


def comm_material_to_dict(material: object) -> dict[str, Any]:
    audio_info = getattr(material, "audio_info", None)
    meta_data = getattr(audio_info, "meta_data", None) if audio_info else None
    return {
        "id": getattr(material, "id", ""),
        "category_code": getattr(material, "category_code", 0),
        "level": getattr(material, "level", 0),
        "noise_color": getattr(material, "noise_color", ""),
        "name": getattr(material, "name", ""),
        "description": getattr(material, "description", ""),
        "tags": list(getattr(material, "tags", [])),
        "status": getattr(material, "status", 0),
        "create_time": getattr(material, "create_time", ""),
        "update_time": getattr(material, "update_time", ""),
        "audio_info": {
            "meta_data": {
                "url": getattr(meta_data, "url", "") if meta_data else "",
                "duration_sec": getattr(meta_data, "duration_sec", 0) if meta_data else 0,
            },
            "is_loopable": getattr(audio_info, "is_loopable", False) if audio_info else False,
            "is_voice": getattr(audio_info, "is_voice", False) if audio_info else False,
        },
    }


def backup_source_materials(materials: list[object], settings: Settings) -> None:
    backup_path = settings.sync_backup_path
    if backup_path.is_file():
        backup_path.unlink()
        logger.info("已删除上一份原料备份：{}", backup_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [comm_material_to_dict(m) for m in materials]
    backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已备份 {} 条原料至 {}", len(payload), backup_path)


# ---------------------------------------------------------------------------
# 差异同步任务
# ---------------------------------------------------------------------------


class AudioEsSyncJob:
    def __init__(
        self,
        comm: CommClient,
        es_search: EsSearch,
        es_sync: EsSync,
        settings: Settings,
    ) -> None:
        self._comm = comm
        self._es_search = es_search
        self._es_sync = es_sync
        self._settings = settings

    async def run(self, *, dry_run: bool = False) -> SyncJobResult:
        page_size = self._settings.sync_page_size
        deleted = created = updated = unchanged = failed = 0

        logger.info(
            "开始 ES 差异同步（源库为准），page_size={}，dry_run={}",
            page_size,
            dry_run,
        )

        materials = await self._fetch_all_materials(page_size)
        total_fetched = len(materials)

        if not dry_run:
            backup_source_materials(materials, self._settings)
        else:
            logger.info("dry_run 模式：跳过本地原料备份")

        build_result = build_payloads_from_materials(materials)
        payloads = build_result.payloads
        source_ids = set(payloads.keys())
        es_ids = await self._es_search.list_all_audio_doc_ids()

        for doc_id in es_ids - source_ids:
            if dry_run:
                deleted += 1
                continue
            try:
                await self._es_sync.delete_audio(doc_id)
                deleted += 1
            except Exception as exc:
                failed += 1
                logger.error("删除 ES 孤儿文档失败，id={}，原因：{}", doc_id, exc)

        for doc_id, payload in payloads.items():
            outcome = await self._sync_one(doc_id, payload, dry_run=dry_run)
            if outcome == "created":
                created += 1
            elif outcome == "updated":
                updated += 1
            elif outcome == "unchanged":
                unchanged += 1
            elif outcome == "failed":
                failed += 1

        result = SyncJobResult(
            total_fetched=total_fetched,
            unique_sources=build_result.unique_count,
            skipped_invalid=build_result.skipped_invalid,
            deduped=build_result.deduped,
            deleted=deleted,
            created=created,
            updated=updated,
            unchanged=unchanged,
            failed=failed,
        )
        logger.info(
            "ES 差异同步结束：拉取={} 唯一={} 无效跳过={} 重复去重={} "
            "删={} 增={} 改={} 未变={} 失败={} dry_run={}",
            result.total_fetched,
            result.unique_sources,
            result.skipped_invalid,
            result.deduped,
            result.deleted,
            result.created,
            result.updated,
            result.unchanged,
            result.failed,
            dry_run,
        )
        return result

    async def _fetch_all_materials(self, page_size: int) -> list[object]:
        all_materials: list[object] = []
        page = 1
        total = 0
        while True:
            materials, total = await self._comm.list_audio_materials_page(
                page=page,
                page_size=page_size,
            )
            if not materials:
                break
            all_materials.extend(materials)
            logger.info("拉取源库进度 {}/{}", len(all_materials), total)
            if page * page_size >= total:
                break
            page += 1
        return all_materials

    async def _sync_one(self, doc_id: str, payload: EsSyncPayload, *, dry_run: bool) -> str:
        desired = build_compare_snapshot(
            audio_url=payload.audio_url,
            audio_name=payload.audio_name,
            flat_tags=payload.flat_tags,
            evidence_level=payload.evidence_level,
            recommend_weight=payload.recommend_weight,
        )
        existing = await self._es_search.get_audio_source(doc_id)
        if existing is not None and not audio_documents_differ(desired, existing):
            return "unchanged"
        if dry_run:
            return "created" if existing is None else "updated"
        try:
            await self._es_sync.upsert_audio(
                doc_id,
                audio_url=payload.audio_url,
                audio_name=payload.audio_name,
                flat_tags=payload.flat_tags,
                evidence_level=payload.evidence_level,
                recommend_weight=payload.recommend_weight,
            )
            return "created" if existing is None else "updated"
        except Exception as exc:
            logger.error("同步失败，id={}，name={}，原因：{}", doc_id, payload.audio_name, exc)
            return "failed"


# ---------------------------------------------------------------------------
# 服务内定时调度（main.py lifespan 调用）
# ---------------------------------------------------------------------------


async def run_scheduled_sync(state: AppState, settings: Settings) -> None:
    if not state.comm_client or not state.es_search or not state.es_sync:
        logger.error("定时同步跳过：comm / ES 依赖未就绪")
        return
    job = AudioEsSyncJob(state.comm_client, state.es_search, state.es_sync, settings)
    await job.run()


def start_sync_scheduler(state: AppState, settings: Settings) -> None:
    global _scheduler
    if not settings.sync_enabled:
        logger.info("ES 定时同步未启用（SYNC_ENABLED=false）")
        return
    if settings.app_debug:
        logger.info("调试模式跳过 ES 定时同步调度")
        return

    async def _job() -> None:
        logger.info("定时任务触发：comm → ES 差异同步")
        await run_scheduled_sync(state, settings)

    _scheduler = AsyncIOScheduler(timezone=UTC)
    first_run = datetime.now(UTC) + timedelta(days=settings.sync_interval_days)
    _scheduler.add_job(
        _job,
        trigger="interval",
        days=settings.sync_interval_days,
        id="audio_es_sync",
        replace_existing=True,
        next_run_time=first_run,
    )
    _scheduler.start()
    logger.info(
        "ES 定时同步已注册，间隔 {} 天，首次执行 {}",
        settings.sync_interval_days,
        first_run.isoformat(),
    )


def shutdown_sync_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


async def _run_cli(*, dry_run: bool) -> int:
    settings = get_settings()
    setup_logging(settings)
    es_client = AsyncElasticsearch(settings.es_node)
    encoder = Encoder(settings)
    encoder.load()
    comm_client = CommClient(settings)
    await comm_client.connect()
    try:
        es_search = EsSearch(es_client, settings)
        await es_search.ensure_indices()
        es_sync = EsSync(es_client, encoder, settings)
        job = AudioEsSyncJob(comm_client, es_search, es_sync, settings)
        result = await job.run(dry_run=dry_run)
    finally:
        await comm_client.close()
        await es_client.close()
    return 1 if result.failed > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="comm 原料差异同步至 ES")
    parser.add_argument("--dry-run", action="store_true", help="只拉取比对，不写 ES、不备份")
    args = parser.parse_args()
    exit_code = asyncio.run(_run_cli(dry_run=args.dry_run))
    if exit_code != 0:
        logger.error("同步未完全成功，退出码 {}", exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
