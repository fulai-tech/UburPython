#!/usr/bin/env python3
"""重建量产 GetAudio 目录相关 Redis 缓存（原料骨架 + 各语言表）。

只写/清 `somni:audio:catalog:*` 与 `somni:audio:locale:*`。
热榜 `somni:audio:hot:*` 不动。

依赖：SOMNI_REDIS_URL、量产 ES（SOMNI_ES_NODE / somni_es_*）。
需已同步过 somni_audio_materials 与 somni_audio_material_locales。

用法:
  .venv/bin/python scripts/rebuild_somni_audio_catalog_cache.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.somni_redis import create_somni_redis  # noqa: E402
from app.es.client import create_es_client  # noqa: E402
from app.es.search import EsSearch  # noqa: E402
from app.server.somni.audio.catalog import AudioCatalogService  # noqa: E402


async def _run() -> int:
    settings = get_settings()
    setup_logging(settings)
    redis = await create_somni_redis(settings)
    if redis is None:
        logger.error("未配置 SOMNI_REDIS_URL，无法重建目录缓存")
        return 1

    es_client = create_es_client(settings, node=settings.effective_somni_es_node)
    try:
        es_search = EsSearch(
            es_client,
            settings,
            audio_index=settings.somni_es_audio_index,
            tag_dictionary_index=settings.somni_es_tag_vectors_index,
            index_profile="somni",
        )
        await es_search.refresh_audio_index()
        service = AudioCatalogService(None, settings, es_search=es_search, redis=redis)
        count = await service.warm_audio_catalog()
        logger.info("量产目录缓存重建完成：原料={}", count)
        return 0
    except Exception as exc:
        logger.error("重建量产目录缓存失败：{}", exc)
        return 1
    finally:
        await redis.aclose()
        await es_client.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
