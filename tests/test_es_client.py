"""ES 客户端工厂单元测试。"""

from __future__ import annotations

from unittest.mock import patch

from app.core.config import Settings
from app.es.client import create_es_client


def test_create_es_client_uses_pool_and_timeout_settings() -> None:
    """工厂应把连接池与超时配置传给 AsyncElasticsearch。"""
    settings = Settings(
        es_node="http://es.example:9200",
        es_connections_per_node=80,
        es_request_timeout_sec=45.0,
    )
    with patch("app.es.client.AsyncElasticsearch") as mock_cls:
        client = create_es_client(settings)

    mock_cls.assert_called_once_with(
        "http://es.example:9200",
        connections_per_node=80,
        request_timeout=45.0,
        retry_on_timeout=True,
        max_retries=2,
    )
    assert client is mock_cls.return_value
