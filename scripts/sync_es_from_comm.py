#!/usr/bin/env python3
"""Mongo Somni 集合 → ES 差异同步（单文件：适配、备份、对账、向量化、定时调度）。

以 Mongo _id 为准，只读不写源库：
  - ES 有、Mongo 无 → 删 ES
  - Mongo 有 → 比差异，有变才 upsert
  - 先同步 somni_audio_tag_dictionary（name/name_en 向量），再同步 somni_audio_materials

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
from typing import TYPE_CHECKING, Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from bson import ObjectId  # noqa: E402
from elasticsearch import AsyncElasticsearch  # noqa: E402
from loguru import logger  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.embedding.encoder import Encoder, create_encoder  # noqa: E402
from app.es.search import EsSearch  # noqa: E402

if TYPE_CHECKING:
    from app.main import AppState

_scheduler: AsyncIOScheduler | None = None
TAG_STATUS_ACTIVE = "启用"
MATERIAL_STATUS_ACTIVE = True
DESCRIPTION_TAG_FIELDS = (
    "sleep_stage_tags",
    "content_form_tags",
    "mechanism_tags",
    "audio_engineering_tags",
    "medical_risk_tags",
    "evidence_level_tags",
)


@dataclass(frozen=True)
class SyncJobResult:
    tag_fetched: int = 0
    tag_deleted: int = 0
    tag_created: int = 0
    tag_updated: int = 0
    tag_unchanged: int = 0
    tag_failed: int = 0
    material_fetched: int = 0
    material_skipped: int = 0
    material_deleted: int = 0
    material_created: int = 0
    material_updated: int = 0
    material_unchanged: int = 0
    material_failed: int = 0

    @property
    def failed(self) -> int:
        return self.tag_failed + self.material_failed


def bson_to_jsonable(value: Any) -> Any:
    """BSON 值 → JSON/ES 可序列化类型。"""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: bson_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bson_to_jsonable(v) for v in value]
    return value


def mongo_doc_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id", "")).strip()


def material_doc_to_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Mongo 原料文档 → ES 文档（去掉 _id）。"""
    doc_id = mongo_doc_id(doc)
    audio_url = str(doc.get("audio_url", "")).strip()
    if not doc_id or not audio_url:
        return None
    payload = bson_to_jsonable(doc)
    payload.pop("_id", None)
    payload["description_text"] = build_material_description_text(payload)
    return payload


def build_material_description_text(doc: dict[str, Any]) -> str:
    labels: list[str] = []
    for field in DESCRIPTION_TAG_FIELDS:
        items = doc.get(field) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                labels.extend(
                    str(item.get(key, "")).strip()
                    for key in ("name", "code", "en_name")
                    if item.get(key)
                )
    parts = [
        str(doc.get("audio_name", "")).strip(),
        str(doc.get("description", "")).strip(),
        " ".join(label for label in labels if label).strip(),
    ]
    return " ".join(part for part in parts if part)


def tag_dictionary_compare_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    """标签词典 diff 快照（不含向量）。"""
    keys = (
        "type",
        "code",
        "status",
        "name",
        "name_en",
        "description",
        "applicability",
        "parent_tag_id",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
    )
    return {k: doc.get(k) for k in keys}


def material_compare_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    """原料 diff 快照（不比较 dense vector，避免浮点噪声导致反复更新）。"""
    snapshot = bson_to_jsonable(doc)
    snapshot.pop("_id", None)
    snapshot.pop("id", None)
    snapshot.pop("description_vector", None)
    return snapshot


def tag_documents_differ(desired: dict[str, Any], existing: dict[str, Any]) -> bool:
    return tag_dictionary_compare_snapshot(desired) != tag_dictionary_compare_snapshot(existing)


def material_documents_differ(desired: dict[str, Any], existing: dict[str, Any]) -> bool:
    return material_compare_snapshot(desired) != material_compare_snapshot(existing)


def zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def write_backup(path: Path, records: list[dict[str, Any]]) -> None:
    if path.is_file():
        path.unlink()
        logger.info("已删除上一份备份：{}", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已备份 {} 条记录至 {}", len(records), path)


class MongoSource:
    """MongoDB Somni 集合只读访问。"""

    def __init__(self, settings: Settings) -> None:
        if not settings.mongo_uri:
            msg = "MONGO_URI 未配置，无法连接 MongoDB"
            raise ValueError(msg)
        self._client = AsyncIOMotorClient(settings.mongo_uri)
        self._db = self._client[settings.mongo_db]
        self._materials = settings.mongo_materials_collection
        self._dictionary = settings.mongo_tag_dictionary_collection
        self._page_size = settings.sync_page_size

    async def close(self) -> None:
        self._client.close()

    async def fetch_tag_dictionary(self) -> list[dict[str, Any]]:
        return await self._fetch_active(self._dictionary, TAG_STATUS_ACTIVE)

    async def fetch_materials(self) -> list[dict[str, Any]]:
        return await self._fetch_active_materials()

    async def _fetch_active(self, collection: str, status: str) -> list[dict[str, Any]]:
        coll = self._db[collection]
        cursor = coll.find({"status": status})
        return [doc async for doc in cursor]

    async def _fetch_active_materials(self) -> list[dict[str, Any]]:
        coll = self._db[self._materials]
        cursor = coll.find({"status": MATERIAL_STATUS_ACTIVE})
        return [doc async for doc in cursor]


class TagDictionarySyncJob:
    def __init__(
        self,
        mongo: MongoSource,
        es_search: EsSearch,
        es_client: AsyncElasticsearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._mongo = mongo
        self._es_search = es_search
        self._client = es_client
        self._encoder = encoder
        self._settings = settings
        self._dim = settings.embedding_dim

    async def run(self, *, dry_run: bool) -> dict[str, int]:
        stats = {
            "fetched": 0,
            "deleted": 0,
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        }
        docs = await self._mongo.fetch_tag_dictionary()
        stats["fetched"] = len(docs)

        if not dry_run:
            write_backup(self._settings.sync_tag_dictionary_backup_path, bson_to_jsonable(docs))

        source_ids = {mongo_doc_id(d) for d in docs if mongo_doc_id(d)}
        es_ids = await self._es_search.list_all_tag_dictionary_doc_ids()

        for doc_id in es_ids - source_ids:
            if dry_run:
                stats["deleted"] += 1
                continue
            try:
                await self._client.delete(index=self._es_search.tag_dictionary_index, id=doc_id)
                stats["deleted"] += 1
            except Exception as exc:
                stats["failed"] += 1
                logger.error("删除 ES 孤儿标签失败，id={}，原因：{}", doc_id, exc)

        for doc in docs:
            outcome = await self._sync_one(doc, dry_run=dry_run)
            stats[outcome] += 1

        return stats

    async def _sync_one(self, doc: dict[str, Any], *, dry_run: bool) -> str:
        doc_id = mongo_doc_id(doc)
        if not doc_id:
            return "failed"
        es_doc = bson_to_jsonable(doc)
        es_doc.pop("_id", None)
        existing = await self._es_search.get_tag_dictionary_source(doc_id)
        if existing and not tag_documents_differ(es_doc, existing):
            return "unchanged"
        if dry_run:
            return "created" if existing is None else "updated"
        try:
            es_doc["name_vector"] = await self._encoder.encode_one(str(es_doc.get("name", "")))
            name_en = str(es_doc.get("name_en", "")).strip()
            es_doc["name_en_vector"] = (
                await self._encoder.encode_one(name_en) if name_en else zero_vector(self._dim)
            )
            await self._client.index(
                index=self._es_search.tag_dictionary_index,
                id=doc_id,
                document=es_doc,
            )
            return "created" if existing is None else "updated"
        except Exception as exc:
            logger.error(
                "同步标签词典失败，id={}，name={}，原因：{}",
                doc_id,
                es_doc.get("name"),
                exc,
            )
            return "failed"


class MaterialsSyncJob:
    def __init__(
        self,
        mongo: MongoSource,
        es_search: EsSearch,
        es_client: AsyncElasticsearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._mongo = mongo
        self._es_search = es_search
        self._client = es_client
        self._encoder = encoder
        self._settings = settings

    async def run(self, *, dry_run: bool) -> dict[str, int]:
        stats = {
            "fetched": 0,
            "skipped": 0,
            "deleted": 0,
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        }
        docs = await self._mongo.fetch_materials()
        stats["fetched"] = len(docs)

        if not dry_run:
            write_backup(self._settings.sync_backup_path, bson_to_jsonable(docs))

        payloads: dict[str, dict[str, Any]] = {}
        for doc in docs:
            es_doc = material_doc_to_es(doc)
            if es_doc is None:
                stats["skipped"] += 1
                logger.warning("跳过无效原料，id={}，audio_url 缺失", mongo_doc_id(doc))
                continue
            payloads[mongo_doc_id(doc)] = es_doc

        es_ids = await self._es_search.list_all_audio_doc_ids()
        for doc_id in es_ids - set(payloads.keys()):
            if dry_run:
                stats["deleted"] += 1
                continue
            try:
                await self._client.delete(index=self._es_search.audio_index, id=doc_id)
                stats["deleted"] += 1
            except Exception as exc:
                stats["failed"] += 1
                logger.error("删除 ES 孤儿原料失败，id={}，原因：{}", doc_id, exc)

        for doc_id, es_doc in payloads.items():
            outcome = await self._sync_one(doc_id, es_doc, dry_run=dry_run)
            stats[outcome] += 1

        return stats

    async def _sync_one(self, doc_id: str, es_doc: dict[str, Any], *, dry_run: bool) -> str:
        existing = await self._es_search.get_audio_source(doc_id)
        if (
            existing
            and not material_documents_differ(es_doc, existing)
            and existing.get("description_vector")
        ):
            return "unchanged"
        if dry_run:
            return "created" if existing is None else "updated"
        try:
            es_doc["description_vector"] = await self._encoder.encode_one(
                str(es_doc.get("description_text", ""))
            )
            await self._client.index(index=self._es_search.audio_index, id=doc_id, document=es_doc)
            return "created" if existing is None else "updated"
        except Exception as exc:
            logger.error(
                "同步原料失败，id={}，name={}，原因：{}",
                doc_id,
                es_doc.get("audio_name"),
                exc,
            )
            return "failed"


class MongoEsSyncJob:
    """编排标签词典 + 原料双集合同步。"""

    def __init__(
        self,
        mongo: MongoSource,
        es_search: EsSearch,
        es_client: AsyncElasticsearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._mongo = mongo
        self._es_search = es_search
        self._client = es_client
        self._encoder = encoder
        self._settings = settings

    async def run(self, *, dry_run: bool = False) -> SyncJobResult:
        logger.info("开始 Mongo → ES 差异同步，dry_run={}", dry_run)
        await self._es_search.migrate_legacy_indices()
        await self._es_search.ensure_indices()

        tag_job = TagDictionarySyncJob(
            self._mongo, self._es_search, self._client, self._encoder, self._settings
        )
        tag_stats = await tag_job.run(dry_run=dry_run)

        material_job = MaterialsSyncJob(
            self._mongo,
            self._es_search,
            self._client,
            self._encoder,
            self._settings,
        )
        material_stats = await material_job.run(dry_run=dry_run)

        result = SyncJobResult(
            tag_fetched=tag_stats["fetched"],
            tag_deleted=tag_stats["deleted"],
            tag_created=tag_stats["created"],
            tag_updated=tag_stats["updated"],
            tag_unchanged=tag_stats["unchanged"],
            tag_failed=tag_stats["failed"],
            material_fetched=material_stats["fetched"],
            material_skipped=material_stats["skipped"],
            material_deleted=material_stats["deleted"],
            material_created=material_stats["created"],
            material_updated=material_stats["updated"],
            material_unchanged=material_stats["unchanged"],
            material_failed=material_stats["failed"],
        )
        logger.info(
            "Mongo → ES 同步结束：标签 拉取={} 删={} 增={} 改={} 未变={} 失败={}；"
            "原料 拉取={} 跳过={} 删={} 增={} 改={} 未变={} 失败={} dry_run={}",
            result.tag_fetched,
            result.tag_deleted,
            result.tag_created,
            result.tag_updated,
            result.tag_unchanged,
            result.tag_failed,
            result.material_fetched,
            result.material_skipped,
            result.material_deleted,
            result.material_created,
            result.material_updated,
            result.material_unchanged,
            result.material_failed,
            dry_run,
        )
        return result


async def run_scheduled_sync(state: AppState, settings: Settings) -> None:
    if not state.es_search or not state.encoder:
        logger.error("定时同步跳过：ES / Encoder 依赖未就绪")
        return
    if not settings.mongo_uri:
        logger.error("定时同步跳过：MONGO_URI 未配置")
        return
    mongo = MongoSource(settings)
    es_client = state.es_client
    if es_client is None:
        logger.error("定时同步跳过：ES 客户端未就绪")
        return
    job = MongoEsSyncJob(mongo, state.es_search, es_client, state.encoder, settings)
    try:
        await job.run()
    finally:
        await mongo.close()


def start_sync_scheduler(state: AppState, settings: Settings) -> None:
    global _scheduler
    if not settings.sync_enabled:
        logger.info("ES 定时同步未启用（SYNC_ENABLED=false）")
        return
    if settings.app_debug:
        logger.info("调试模式跳过 ES 定时同步调度")
        return
    if not settings.mongo_uri:
        logger.info("MONGO_URI 未配置，跳过 ES 定时同步调度")
        return

    async def _job() -> None:
        logger.info("定时任务触发：Mongo → ES 差异同步")
        await run_scheduled_sync(state, settings)

    _scheduler = AsyncIOScheduler(timezone=UTC)
    first_run = datetime.now(UTC) + timedelta(days=settings.sync_interval_days)
    _scheduler.add_job(
        _job,
        trigger="interval",
        days=settings.sync_interval_days,
        id="mongo_es_sync",
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


async def _run_cli(*, dry_run: bool) -> int:
    settings = get_settings()
    setup_logging(settings)
    es_client = AsyncElasticsearch(settings.es_node)
    encoder = create_encoder(settings)
    encoder.load()
    mongo = MongoSource(settings)
    try:
        es_search = EsSearch(es_client, settings)
        job = MongoEsSyncJob(mongo, es_search, es_client, encoder, settings)
        result = await job.run(dry_run=dry_run)
    finally:
        await mongo.close()
        await es_client.close()
    return 1 if result.failed > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mongo Somni 集合差异同步至 ES")
    parser.add_argument("--dry-run", action="store_true", help="只拉取比对，不写 ES、不备份")
    args = parser.parse_args()
    exit_code = asyncio.run(_run_cli(dry_run=args.dry_run))
    if exit_code != 0:
        logger.error("同步未完全成功，退出码 {}", exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
