"""Elasticsearch 异步客户端工厂（连接池与超时集中配置）。"""

from __future__ import annotations

from elasticsearch import AsyncElasticsearch

from app.core.config import Settings


def create_es_client(settings: Settings) -> AsyncElasticsearch:
    """创建进程级 ES 客户端；高并发检索需足够 connections_per_node。"""
    return AsyncElasticsearch(
        settings.es_node,
        connections_per_node=settings.es_connections_per_node,
        request_timeout=settings.es_request_timeout_sec,
        retry_on_timeout=True,
        max_retries=2,
    )
