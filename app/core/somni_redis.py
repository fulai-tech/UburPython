"""量产 Redis 客户端（与功能手板/HTTP Redis 物理隔离）。"""

from __future__ import annotations

from loguru import logger
from redis.asyncio import Redis

from app.core.config import Settings


def resolve_somni_redis_url(settings: Settings) -> str:
    return settings.somni_redis_url.strip()


async def create_somni_redis(settings: Settings) -> Redis | None:
    """按 SOMNI_REDIS_URL 建连接，供 GetHot 与 GetAudio 目录缓存共用。空 URL 不回退手板 Redis。"""
    url = resolve_somni_redis_url(settings)
    if not url:
        logger.warning("未配置 SOMNI_REDIS_URL，量产 GetHot 与 GetAudio 目录缓存不可用")
        return None
    client = Redis.from_url(
        url,
        decode_responses=True,
        max_connections=max(1, settings.somni_redis_max_connections),
        socket_connect_timeout=max(0.1, settings.somni_redis_connect_timeout_sec),
        socket_timeout=max(0.1, settings.somni_redis_socket_timeout_sec),
        health_check_interval=30,
    )
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        raise
    logger.info(
        "已连接量产独立 Redis，max_connections={}",
        settings.somni_redis_max_connections,
    )
    return client
