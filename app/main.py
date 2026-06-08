"""FastAPI 应用入口。

职责：
- lifespan 内单例预热 ES / Embedding / gRPC 客户端（规范要求勿每请求重建）
- 挂载 HTTP 路由与请求日志中间件
- 通过 AppState 向 API 层提供已初始化的服务实例
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from loguru import logger

from app.api.audio import router as audio_router
from app.bionode_grpc_clients import CommClient
from app.core.config import Settings, get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.es.sync import EsSync
from app.middleware.request_log import register_request_log_middleware
from app.services.audio import AudioService
from app.services.retrieval import RetrievalService


@dataclass
class AppState:
    """进程级单例容器，在 lifespan 中填充，供依赖注入读取。"""

    settings: Settings
    es_client: AsyncElasticsearch | None = None
    encoder: Encoder | None = None
    comm_client: CommClient | None = None
    es_search: EsSearch | None = None
    es_sync: EsSync | None = None
    retrieval_service: RetrievalService | None = None
    audio_service: AudioService | None = None


_app_state = AppState(settings=get_settings())


def get_app_state() -> AppState:
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings)
    _app_state.settings = settings

    logger.info("正在启动 UburNode 音频检索服务")

    es_client = AsyncElasticsearch(settings.es_node)
    _app_state.es_client = es_client

    encoder = Encoder(settings)
    # debug 模式跳过向量模型加载：本地无外网时可先调 HTTP 路由
    if not settings.app_debug:
        encoder.load()
    _app_state.encoder = encoder

    comm_client = CommClient(settings)
    await comm_client.connect()
    _app_state.comm_client = comm_client

    es_search = EsSearch(es_client, settings)
    try:
        await es_search.ensure_indices()
    except Exception as exc:
        # 生产环境 ES 不可达应 fail fast；debug 允许仅验证 API 层
        if settings.app_debug:
            logger.warning("Elasticsearch 不可用（调试模式，继续启动）：{}", exc)
        else:
            raise
    _app_state.es_search = es_search

    es_sync = EsSync(es_client, encoder, settings)
    _app_state.es_sync = es_sync

    retrieval = RetrievalService(es_search, encoder, settings)
    _app_state.retrieval_service = retrieval

    _app_state.audio_service = AudioService(comm_client, es_sync, retrieval)

    logger.info("UburNode 音频检索服务已就绪")
    yield

    logger.info("正在关闭 UburNode 音频检索服务")
    await comm_client.close()
    await es_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="UburNode Audio Search Service",
        description="音频检索服务 — HTTP 对外，gRPC 调 comm-service，ES 索引副本",
        version="0.1.0",
        lifespan=lifespan,
        debug=False,
    )
    register_exception_handlers(app)
    register_request_log_middleware(app)
    app.include_router(audio_router, prefix="/api")

    from app.schemas.response import ApiResponse, success

    @app.get("/health", response_model=ApiResponse)
    async def health() -> ApiResponse:
        return success(data={}, msg="服务正常")

    return app


app = create_app()
