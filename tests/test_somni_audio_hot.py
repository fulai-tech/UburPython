import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.es.search_events import SearchEventsStore


def test_resolve_somni_redis_url_empty_does_not_fall_back() -> None:
    from app.core.somni_redis import resolve_somni_redis_url

    settings = Settings(somni_redis_url="   ", redis_url="redis://localhost:6379/0")
    assert resolve_somni_redis_url(settings) == ""


def test_resolve_somni_redis_url_prefers_somni() -> None:
    from app.core.somni_redis import resolve_somni_redis_url

    settings = Settings(
        somni_redis_url="redis://somni:6379/1",
        redis_url="redis://localhost:6379/0",
    )
    assert resolve_somni_redis_url(settings) == "redis://somni:6379/1"


@pytest.mark.asyncio
async def test_create_somni_redis_uses_only_somni_config(monkeypatch) -> None:
    from app.core.somni_redis import create_somni_redis

    client = MagicMock()
    client.ping = AsyncMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("app.core.somni_redis.Redis.from_url", factory)
    settings = Settings(
        somni_redis_url="redis://somni:6379/0",
        redis_url="redis://shared:6379/0",
        somni_redis_max_connections=64,
        somni_redis_connect_timeout_sec=1.5,
        somni_redis_socket_timeout_sec=2.5,
    )

    assert await create_somni_redis(settings) is client

    factory.assert_called_once_with(
        "redis://somni:6379/0",
        decode_responses=True,
        max_connections=64,
        socket_connect_timeout=1.5,
        socket_timeout=2.5,
        health_check_interval=30,
    )
    client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_somni_redis_closes_failed_client(monkeypatch) -> None:
    from app.core.somni_redis import create_somni_redis

    client = MagicMock()
    client.ping = AsyncMock(side_effect=ConnectionError("unavailable"))
    client.aclose = AsyncMock()
    monkeypatch.setattr("app.core.somni_redis.Redis.from_url", MagicMock(return_value=client))

    with pytest.raises(ConnectionError):
        await create_somni_redis(Settings(somni_redis_url="redis://somni:6379/0"))

    client.aclose.assert_awaited_once()


def test_hot_settings_defaults() -> None:
    s = Settings()
    assert s.somni_hot_enabled is True
    assert s.somni_hot_top_n == 10
    assert s.somni_hot_redis_key == "somni:audio:hot:v1"
    assert s.somni_redis_max_connections == 128
    assert s.somni_redis_connect_timeout_sec == 2.0
    assert s.somni_redis_socket_timeout_sec == 2.0
    assert s.somni_es_search_events_index == "somni_audio_search_events"


@pytest.mark.asyncio
async def test_search_events_ensure_and_index() -> None:
    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.index = AsyncMock()
    store = SearchEventsStore(client, Settings())
    await store.ensure_index()
    client.indices.create.assert_awaited()
    await store.index_event(keyword="雨声", raw_query=" 雨声 ", hit_count=2)
    client.indices.exists.assert_awaited_once()
    client.indices.create.assert_awaited_once()
    kwargs = client.index.await_args.kwargs
    assert kwargs["index"] == "somni_audio_search_events"
    assert kwargs["document"]["keyword"] == "雨声"
    assert kwargs["document"]["hit_count"] == 2


@pytest.mark.asyncio
async def test_search_events_already_exists_still_indexes_and_caches_ensure() -> None:
    class _AlreadyExistsError(Exception):
        error = "resource_already_exists_exception"

    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock(side_effect=_AlreadyExistsError())
    client.index = AsyncMock()
    store = SearchEventsStore(client, Settings())

    await asyncio.gather(
        store.index_event(keyword="雨声", raw_query=" 雨声 ", hit_count=1),
        store.index_event(keyword="风声", raw_query="风声", hit_count=2),
    )

    client.indices.exists.assert_awaited_once()
    client.indices.create.assert_awaited_once()
    assert client.index.await_count == 2


def test_normalize_keyword_strips() -> None:
    from app.server.somni.audio.hot import normalize_keyword

    assert normalize_keyword(" 雨声 ") == "雨声"
    assert normalize_keyword("   ") == ""


@pytest.mark.asyncio
async def test_record_and_list_hot() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock()
    redis.zrevrange = AsyncMock(
        return_value=[("雨声".encode(), 2.0), ("暴雨声".encode(), 1.0)]
    )
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())
    await tracker.record_search(" 雨声 ", hit_count=3)
    redis.zincrby.assert_awaited_once()
    events.index_event.assert_awaited_once()
    items = await tracker.list_hot()
    assert items == [{"keyword": "雨声", "score": 2}, {"keyword": "暴雨声", "score": 1}]


@pytest.mark.asyncio
async def test_record_blank_skipped() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())
    await tracker.record_search("  ", hit_count=0)
    redis.zincrby.assert_not_called()
    events.index_event.assert_not_called()


@pytest.mark.asyncio
async def test_record_search_redis_failure_still_indexes_es() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock(side_effect=RuntimeError("redis unavailable"))
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search(" 雨声 ", hit_count=3)

    events.index_event.assert_awaited_once_with(
        keyword="雨声",
        raw_query=" 雨声 ",
        hit_count=3,
    )


@pytest.mark.asyncio
async def test_record_search_es_failure_returns_after_redis_increment() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock(side_effect=RuntimeError("es unavailable"))
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search("雨声", hit_count=1)

    redis.zincrby.assert_awaited_once()


@pytest.mark.asyncio
async def test_hot_disabled_skips_writes_and_returns_empty_list() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock()
    redis.zrevrange = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings(somni_hot_enabled=False))

    await tracker.record_search("雨声", hit_count=1)
    assert await tracker.list_hot() == []

    redis.zincrby.assert_not_called()
    redis.zrevrange.assert_not_called()
    events.index_event.assert_not_called()


@pytest.mark.asyncio
async def test_list_hot_non_positive_top_n_returns_empty_without_redis_read() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zrevrange = AsyncMock()
    tracker = HotTracker(redis, None, Settings(somni_hot_top_n=0))

    assert await tracker.list_hot() == []
    redis.zrevrange.assert_not_called()


@pytest.mark.asyncio
async def test_list_hot_requires_redis() -> None:
    from app.core.exceptions import AppError
    from app.server.somni.audio.hot import HotTracker

    tracker = HotTracker(None, None, Settings())
    with pytest.raises(AppError):
        await tracker.list_hot()


@pytest.mark.asyncio
async def test_list_hot_redis_failure_is_service_unavailable() -> None:
    from app.core.exceptions import AppError
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zrevrange = AsyncMock(side_effect=ConnectionError("unavailable"))
    tracker = HotTracker(redis, None, Settings())

    with pytest.raises(AppError) as exc:
        await tracker.list_hot()

    assert exc.value.status_code == 503
