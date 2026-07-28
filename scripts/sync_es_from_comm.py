#!/usr/bin/env python3
"""Mongo Somni 集合 → ES 全量重建同步（单文件：适配、备份、清空、向量化、定时调度）。

以 Mongo `_id` 为准，只读不写源库：
  - 先删除目标 ES 索引全部数据（删索引再 ensure 重建）
  - 再按 Mongo 启用文档全量插入；ES 文档 id = Mongo `_id`，_source 不含 `id`/`_id`
  - 先同步 somni_audio_tag_dictionary，再同步 somni_audio_materials

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
from loguru import logger  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.cache.audio_search_cache import (  # noqa: E402
    create_audio_search_cache,
    shutdown_audio_search_cache,
)
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.embedding.encoder import Encoder, create_encoder  # noqa: E402
from app.es.client import create_es_client  # noqa: E402
from app.es.search import EsSearch  # noqa: E402
from app.es.somni_docs import (  # noqa: E402
    build_material_description_text,
    material_source_for_es,
)
from app.mongo.materials import bson_to_jsonable  # noqa: E402

# 供测试与外部脚本复用
__all__ = (
    "bson_to_jsonable",
    "build_material_description_text",
    "material_doc_to_es",
    "material_source_for_es",
)

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

    from app.main import AppState

_scheduler: AsyncIOScheduler | None = None
TAG_STATUS_ACTIVE = "启用"
MATERIAL_STATUS_ACTIVE = True


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


def mongo_doc_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id", "")).strip()


def material_doc_to_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Mongo 原料文档 → ES _source（去掉 _id/id）；无 _id 或无 audio_url 则跳过。"""
    doc_id = mongo_doc_id(doc)
    if not doc_id:
        return None
    payload = material_source_for_es(bson_to_jsonable(doc))
    if payload is None:
        return None
    payload.pop("_id", None)
    payload.pop("id", None)
    return payload


def tag_doc_to_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Mongo 标签文档 → ES _source（去掉 _id/id）；无 _id 则跳过。"""
    doc_id = mongo_doc_id(doc)
    if not doc_id:
        return None
    payload = bson_to_jsonable(doc)
    payload.pop("_id", None)
    payload.pop("id", None)
    return payload


def zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


async def wipe_and_recreate_index(
    es_client: AsyncElasticsearch,
    es_search: EsSearch,
    index: str,
) -> int:
    """删除索引内全部数据：统计文档数 → 删索引 → ensure 重建映射。"""
    count = 0
    if await es_client.indices.exists(index=index):
        count_resp = await es_client.count(index=index)
        count = int(count_resp.get("count", 0))
        await es_client.indices.delete(index=index)
        logger.info("已删除 ES 索引以全量重建：{}，原文档数={}", index, count)
    await es_search.ensure_indices()
    return count


def write_backup(path: Path, records: list[dict[str, Any]]) -> None:
    if path.is_file():
        path.unlink()
        logger.info("已删除上一份备份：{}", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已备份 {} 条记录至 {}", len(records), path)


def _redact_mongo_uri(uri: str) -> str:
    """日志用：保留 scheme/用户/主机，隐藏密码。"""
    if "://" not in uri or "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    creds, host_part = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0] if ":" in creds else "***"
    return f"{scheme}://{user}:***@{host_part}"


class MongoSource:
    """MongoDB Somni 集合只读访问。"""

    def __init__(self, settings: Settings) -> None:
        if not settings.mongo_uri:
            msg = "MONGO_URI 未配置，无法连接 MongoDB"
            raise ValueError(msg)
        logger.info(
            "同步将连接 MongoDB，MONGO_URI={}，MONGO_DB={}",
            _redact_mongo_uri(settings.mongo_uri),
            settings.mongo_db,
        )
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

        payloads: dict[str, dict[str, Any]] = {}
        for doc in docs:
            es_doc = tag_doc_to_es(doc)
            if es_doc is None:
                stats["failed"] += 1
                continue
            payloads[mongo_doc_id(doc)] = es_doc

        if dry_run:
            es_ids = await self._es_search.list_all_tag_dictionary_doc_ids()
            stats["deleted"] = len(es_ids)
            stats["created"] = len(payloads)
            return stats

        stats["deleted"] = await wipe_and_recreate_index(
            self._client,
            self._es_search,
            self._es_search.tag_dictionary_index,
        )
        for doc_id, es_doc in payloads.items():
            outcome = await self._insert_one(doc_id, es_doc)
            stats[outcome] += 1

        self._es_search.clear_content_tag_vectors_cache()
        return stats

    async def _insert_one(self, doc_id: str, es_doc: dict[str, Any]) -> str:
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
            return "created"
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

        if dry_run:
            es_ids = await self._es_search.list_all_audio_doc_ids()
            stats["deleted"] = len(es_ids)
            stats["created"] = len(payloads)
            return stats

        stats["deleted"] = await wipe_and_recreate_index(
            self._client,
            self._es_search,
            self._es_search.audio_index,
        )
        for doc_id, es_doc in payloads.items():
            outcome = await self._insert_one(doc_id, es_doc)
            stats[outcome] += 1

        return stats

    async def _insert_one(self, doc_id: str, es_doc: dict[str, Any]) -> str:
        try:
            es_doc["description_vector"] = await self._encoder.encode_one(
                str(es_doc.get("description_text", ""))
            )
            await self._client.index(
                index=self._es_search.audio_index,
                id=doc_id,
                document=es_doc,
            )
            return "created"
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
        logger.info("开始 Mongo → ES 全量重建同步，dry_run={}", dry_run)
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
            "Mongo → ES 全量同步结束：标签 拉取={} 清索引={} 增={} 失败={}；"
            "原料 拉取={} 跳过={} 清索引={} 增={} 失败={} dry_run={}",
            result.tag_fetched,
            result.tag_deleted,
            result.tag_created,
            result.tag_failed,
            result.material_fetched,
            result.material_skipped,
            result.material_deleted,
            result.material_created,
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
        result = await job.run()
        await _clear_search_cache_after_material_sync(state, result)
    finally:
        await mongo.close()


async def _clear_search_cache_after_material_sync(state: AppState, result: SyncJobResult) -> None:
    if not (result.material_created or result.material_updated or result.material_deleted):
        return
    if state.search_cache is not None:
        await state.search_cache.clear_all()
    retrieval = state.retrieval_service
    if retrieval is None:
        return
    await retrieval.clear_sleep_stage_cache()
    await retrieval.warm_sleep_stage_cache()


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
        logger.info("定时任务触发：Mongo → ES 全量重建同步")
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
    es_client = create_es_client(settings)
    encoder = create_encoder(settings)
    encoder.load()
    mongo = MongoSource(settings)
    search_cache = None
    try:
        es_search = EsSearch(es_client, settings)
        job = MongoEsSyncJob(mongo, es_search, es_client, encoder, settings)
        result = await job.run(dry_run=dry_run)
        if not dry_run and (
            result.material_created or result.material_updated or result.material_deleted
        ):
            search_cache = await create_audio_search_cache(settings)
            if search_cache is not None:
                await search_cache.clear_all()
            from app.cache.sleep_stage_cache import create_sleep_stage_candidate_cache

            sleep_redis = search_cache.redis if search_cache is not None else None
            sleep_cache = await create_sleep_stage_candidate_cache(settings, redis=sleep_redis)
            if sleep_cache is not None:
                await sleep_cache.clear_all()

                async def _load_stage(stage: str) -> list:
                    return await es_search.filter_by_sleep_stage([stage])

                await sleep_cache.warm(_load_stage)
    finally:
        await shutdown_audio_search_cache(search_cache)
        await mongo.close()
        await es_client.close()
    return 1 if result.failed > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mongo Somni 集合全量重建同步至 ES")
    parser.add_argument("--dry-run", action="store_true", help="只拉取统计，不删 ES、不写 ES、不备份")
    args = parser.parse_args()
    exit_code = asyncio.run(_run_cli(dry_run=args.dry_run))
    if exit_code != 0:
        logger.error("同步未完全成功，退出码 {}", exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
