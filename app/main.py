"""FastAPI 应用入口。

职责：
- lifespan 内预热 ES / Embedding / Mongo；启功能手板 + 量产 gRPC
- 挂载 HTTP 路由与请求日志中间件
- 通过 AppState 向 API 层提供已初始化的服务实例
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch
    from motor.motor_asyncio import AsyncIOMotorClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"


def _bootstrap_dev_entry() -> None:
    if __name__ != "__main__":
        return
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if not _VENV_PYTHON.is_file():
        return
    if Path(sys.executable).resolve() == _VENV_PYTHON.resolve():
        return
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), *sys.argv])


_bootstrap_dev_entry()

from fastapi import FastAPI
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient

from app.api.audio import router as audio_router
from app.cache.audio_search_cache import (
    AudioSearchCache,
    create_audio_search_cache,
    shutdown_audio_search_cache,
)
from app.cache.sleep_stage_cache import (
    SleepStageCandidateCache,
    create_sleep_stage_candidate_cache,
)
from app.core.config import Settings, get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.somni_redis import create_somni_redis
from app.embedding.encoder import Encoder, create_encoder
from app.es.client import create_es_client
from app.es.search import EsSearch
from app.es.search_events import SearchEventsStore
from app.es.sync import EsSync
from app.middleware.request_log import register_request_log_middleware
from app.server.bootstrap import GrpcServers, start_grpc_servers, stop_grpc_servers
from app.server.handboard.audio.service import AudioService
from app.server.handboard.audio.store import MaterialsStore, create_materials_store
from app.server.somni.audio.catalog import AudioCatalogService as SomniAudioService
from app.server.somni.audio.hot import HotTracker
from app.server.somni.quiz.service import QuizService as SomniQuizService
from app.server.somni.report.service import ReportService as SomniReportService
from app.services.retrieval import RetrievalService
from scripts.sync_es_from_comm import shutdown_sync_scheduler, start_sync_scheduler


@dataclass
class AppState:
    settings: Settings
    es_client: AsyncElasticsearch | None = None
    somni_es_client: AsyncElasticsearch | None = None
    encoder: Encoder | None = None
    materials_store: MaterialsStore | None = None
    somni_mongo_client: AsyncIOMotorClient | None = None
    es_search: EsSearch | None = None
    es_sync: EsSync | None = None
    retrieval_service: RetrievalService | None = None
    audio_service: AudioService | None = None
    somni_quiz_service: SomniQuizService | None = None
    somni_report_service: SomniReportService | None = None
    somni_audio_service: SomniAudioService | None = None
    search_cache: AudioSearchCache | None = None
    sleep_stage_cache: SleepStageCandidateCache | None = None
    grpc_servers: GrpcServers | None = None


_app_state = AppState(settings=get_settings())


def get_app_state() -> AppState:
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings)
    _app_state.settings = settings

    logger.info("正在启动 UburNode 音频检索服务")

    es_client = create_es_client(settings)
    _app_state.es_client = es_client

    encoder = create_encoder(settings)
    if not settings.app_debug:
        encoder.load()
    _app_state.encoder = encoder

    es_search = EsSearch(es_client, settings)
    try:
        await es_search.ensure_indices()
    except Exception as exc:
        if settings.app_debug:
            logger.warning("Elasticsearch 不可用（调试模式，继续启动）：{}", exc)
        else:
            raise
    _app_state.es_search = es_search
    if not settings.app_debug:
        try:
            await es_search.warm_dictionary_vectors_cache()
        except Exception as exc:
            logger.warning("启动预热标签词典向量缓存失败：{}", exc)

    es_sync = EsSync(es_client, encoder, settings)
    _app_state.es_sync = es_sync

    search_cache = await create_audio_search_cache(settings)
    _app_state.search_cache = search_cache

    sleep_redis = search_cache.redis if search_cache is not None else None
    sleep_stage_cache = await create_sleep_stage_candidate_cache(settings, redis=sleep_redis)
    _app_state.sleep_stage_cache = sleep_stage_cache

    retrieval = RetrievalService(
        es_search,
        encoder,
        settings,
        sleep_stage_cache=sleep_stage_cache,
    )
    _app_state.retrieval_service = retrieval
    if not settings.app_debug:
        try:
            await retrieval.warm_query_tag_vectors()
        except Exception as exc:
            logger.warning("启动预热查询标签向量缓存失败：{}", exc)
    if sleep_stage_cache is not None:
        try:
            await retrieval.warm_sleep_stage_cache()
        except Exception as exc:
            logger.warning("启动预热睡眠阶段候选缓存失败：{}", exc)

    materials_store = create_materials_store(settings)
    _app_state.materials_store = materials_store
    if materials_store is None:
        logger.warning("未配置 MONGO_URI，手板写路径将不可用")

    _app_state.audio_service = AudioService(
        materials_store,
        es_sync,
        retrieval,
        search_cache=search_cache,
        sleep_stage_cache=sleep_stage_cache,
    )

    somni_mongo = None
    if settings.somni_mongo_uri:
        somni_mongo = AsyncIOMotorClient(settings.somni_mongo_uri)
        _app_state.somni_mongo_client = somni_mongo
    else:
        logger.warning("未配置 SOMNI_MONGO_URI，量产问卷与音频查询将不可用")

    _app_state.somni_quiz_service = SomniQuizService(somni_mongo, settings)
    _app_state.somni_report_service = SomniReportService()
    somni_es_client = create_es_client(
        settings,
        node=settings.effective_somni_es_node,
    )
    _app_state.somni_es_client = somni_es_client
    somni_es_search = EsSearch(
        somni_es_client,
        settings,
        audio_index=settings.somni_es_audio_index,
        tag_dictionary_index=settings.somni_es_tag_vectors_index,
    )
    somni_redis = await create_somni_redis(settings)
    events_store = SearchEventsStore(somni_es_client, settings)
    hot_tracker = HotTracker(somni_redis, events_store, settings)
    _app_state.somni_audio_service = SomniAudioService(
        somni_mongo,
        settings,
        es_search=somni_es_search,
        encoder=encoder,
        hot=hot_tracker,
    )

    start_sync_scheduler(_app_state, settings)
    _app_state.grpc_servers = await start_grpc_servers(_app_state, settings)

    logger.info("UburNode 音频检索服务已就绪")
    yield

    logger.info("正在关闭 UburNode 音频检索服务")
    await stop_grpc_servers(_app_state.grpc_servers)
    _app_state.grpc_servers = None
    shutdown_sync_scheduler()
    if _app_state.somni_audio_service is not None:
        await _app_state.somni_audio_service.drain_hot_tasks()
    await shutdown_audio_search_cache(search_cache)
    if materials_store is not None:
        materials_store.close()
    if somni_mongo is not None:
        somni_mongo.close()
    if somni_redis is not None:
        await somni_redis.aclose()
    await somni_es_client.close()
    await es_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="UburNode Audio Search Service",
        description="音频检索服务 — HTTP + 功能手板/量产 gRPC；直连 Mongo，ES 为索引副本",
        version="0.1.0",
        lifespan=lifespan,
        debug=False,
    )
    register_exception_handlers(app)
    register_request_log_middleware(app)
    app.include_router(audio_router, prefix="/api")
    return app


app = create_app()


def run_dev_server() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )


if __name__ == "__main__":
    run_dev_server()
