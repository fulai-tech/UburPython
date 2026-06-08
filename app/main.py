"""FastAPI 应用入口。

职责：
- lifespan 内单例预热 ES / Embedding / gRPC 客户端（规范要求勿每请求重建）
- 挂载 HTTP 路由与请求日志中间件
- 通过 AppState 向 API 层提供已初始化的服务实例
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"


def _bootstrap_dev_entry() -> None:
    """直接运行 main.py 时切到项目 .venv，并保证可 import app 包。"""
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
from scripts.sync_es_from_comm import shutdown_sync_scheduler, start_sync_scheduler


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

    start_sync_scheduler(_app_state, settings)

    logger.info("UburNode 音频检索服务已就绪")
    yield

    logger.info("正在关闭 UburNode 音频检索服务")
    shutdown_sync_scheduler()
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
    return app


app = create_app()


def run_dev_server() -> None:
    """本地开发一键启动（读取 .env 的 APP_HOST / APP_PORT）。"""
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
