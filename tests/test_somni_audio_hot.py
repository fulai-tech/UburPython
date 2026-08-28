from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import asyncio
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
    await store.index_event(
        keyword="雨声",
        raw_query=" 雨声 ",
        language="zh",
        hit_count=2,
        kind="query",
    )
    client.indices.exists.assert_awaited_once()
    client.indices.create.assert_awaited_once()
    kwargs = client.index.await_args.kwargs
    assert kwargs["index"] == "somni_audio_search_events"
    assert kwargs["document"]["keyword"] == "雨声"
    assert kwargs["document"]["language"] == "zh"
    assert kwargs["document"]["kind"] == "query"
    assert kwargs["document"]["hit_count"] == 2


@pytest.mark.asyncio
async def test_search_events_existing_index_puts_language_mapping() -> None:
    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.put_mapping = AsyncMock()
    client.indices.create = AsyncMock()
    store = SearchEventsStore(client, Settings())
    await store.ensure_index()
    client.indices.create.assert_not_called()
    client.indices.put_mapping.assert_awaited_once()
    body = client.indices.put_mapping.await_args.kwargs["body"]
    assert body["properties"]["language"]["type"] == "keyword"
    assert body["properties"]["kind"]["type"] == "keyword"


@pytest.mark.asyncio
async def test_search_events_already_exists_still_indexes_and_caches_ensure() -> None:
    class _AlreadyExistsError(Exception):
        error = "resource_already_exists_exception"

    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock(side_effect=_AlreadyExistsError())
    client.indices.put_mapping = AsyncMock()
    client.index = AsyncMock()
    store = SearchEventsStore(client, Settings())

    await asyncio.gather(
        store.index_event(keyword="雨声", raw_query=" 雨声 ", language="zh", hit_count=1),
        store.index_event(keyword="风声", raw_query="风声", language="zh", hit_count=2),
    )

    client.indices.exists.assert_awaited_once()
    client.indices.create.assert_awaited_once()
    assert client.index.await_count == 2


def test_normalize_keyword_and_week_key(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import hot_redis_key, hot_week_id, normalize_keyword

    assert normalize_keyword(" 雨声 ") == "雨声"
    assert normalize_keyword("  Rain   Sound ") == "rain sound"
    assert normalize_keyword("RAIN") == "rain"
    assert normalize_keyword("   ") == ""

    fixed = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)  # ISO 2026-W35
    assert hot_week_id(fixed) == "2026-W35"
    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    assert hot_redis_key(Settings(), "en") == "somni:audio:hot:v1:query:en:2026-W35"
    assert (
        hot_redis_key(Settings(), "zh", kind="tag")
        == "somni:audio:hot:v1:tag:zh:2026-W35"
    )


def test_parse_hot_kind() -> None:
    from app.server.somni.audio.hot import parse_hot_kind

    assert parse_hot_kind(None) == "query"
    assert parse_hot_kind("") == "query"
    assert parse_hot_kind("tag") == "tag"
    assert parse_hot_kind("QUERY") == "query"
    with pytest.raises(ValueError):
        parse_hot_kind("other")


@pytest.mark.asyncio
async def test_record_and_list_hot(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock()
    redis.zrevrange = AsyncMock(
        return_value=[("雨声".encode(), 2.0), ("暴雨声".encode(), 1.0)]
    )
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())
    await tracker.record_search(" 雨声 ", language="zh", hit_count=3)
    redis.zincrby.assert_awaited_once_with(
        "somni:audio:hot:v1:query:zh:2026-W35", 1, "雨声"
    )
    events.index_event.assert_awaited_once_with(
        keyword="雨声",
        raw_query=" 雨声 ",
        language="zh",
        hit_count=3,
        kind="query",
    )
    items = await tracker.list_hot(language="zh")
    assert items == [{"keyword": "雨声", "score": 2}, {"keyword": "暴雨声", "score": 1}]
    redis.zrevrange.assert_awaited_once_with(
        "somni:audio:hot:v1:query:zh:2026-W35", 0, 9, withscores=True
    )


@pytest.mark.asyncio
async def test_record_search_increments_tag_codes(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock()
    redis.zrevrange = AsyncMock(return_value=[("heavy_rain", 4.0)])
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search(
        "rain",
        language="en",
        hit_count=2,
        tag_codes={"Heavy_Rain", "steady_rain"},
    )

    redis.zincrby.assert_any_await(
        "somni:audio:hot:v1:query:en:2026-W35", 1, "rain"
    )
    redis.zincrby.assert_any_await(
        "somni:audio:hot:v1:tag:en:2026-W35", 1, "heavy_rain"
    )
    redis.zincrby.assert_any_await(
        "somni:audio:hot:v1:tag:en:2026-W35", 1, "steady_rain"
    )
    assert events.index_event.await_count == 3
    kinds = {c.kwargs["kind"] for c in events.index_event.await_args_list}
    assert kinds == {"query", "tag"}

    items = await tracker.list_hot(language="en", kind="tag")
    assert items == [{"keyword": "heavy_rain", "score": 4}]
    redis.zrevrange.assert_awaited_once_with(
        "somni:audio:hot:v1:tag:en:2026-W35", 0, 9, withscores=True
    )


@pytest.mark.asyncio
async def test_record_zero_hit_skips_redis_but_indexes_es(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search("rain", language="en", hit_count=0)

    redis.zincrby.assert_not_called()
    events.index_event.assert_awaited_once_with(
        keyword="rain",
        raw_query="rain",
        language="en",
        hit_count=0,
        kind="query",
    )


@pytest.mark.asyncio
async def test_record_blank_skipped() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())
    await tracker.record_search("  ", language="zh", hit_count=0)
    redis.zincrby.assert_not_called()
    events.index_event.assert_not_called()


@pytest.mark.asyncio
async def test_record_search_redis_failure_still_indexes_es(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock(side_effect=RuntimeError("redis unavailable"))
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search(" 雨声 ", language="en", hit_count=3)

    events.index_event.assert_awaited_once_with(
        keyword="雨声",
        raw_query=" 雨声 ",
        language="en",
        hit_count=3,
        kind="query",
    )


@pytest.mark.asyncio
async def test_record_search_normalizes_english_case(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock()
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search("  Rain ", language="en", hit_count=1)

    redis.zincrby.assert_awaited_once_with(
        "somni:audio:hot:v1:query:en:2026-W35", 1, "rain"
    )
    assert events.index_event.await_args.kwargs["keyword"] == "rain"


@pytest.mark.asyncio
async def test_record_search_es_failure_returns_after_redis_increment(monkeypatch) -> None:
    from app.server.somni.audio import hot as hot_mod
    from app.server.somni.audio.hot import HotTracker

    monkeypatch.setattr(hot_mod, "hot_week_id", lambda now=None: "2026-W35")
    redis = MagicMock()
    redis.zincrby = AsyncMock()
    events = MagicMock()
    events.index_event = AsyncMock(side_effect=RuntimeError("es unavailable"))
    tracker = HotTracker(redis, events, Settings())

    await tracker.record_search("雨声", language="zh", hit_count=1)

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

    await tracker.record_search("雨声", language="zh", hit_count=1)
    assert await tracker.list_hot(language="zh") == []

    redis.zincrby.assert_not_called()
    redis.zrevrange.assert_not_called()
    events.index_event.assert_not_called()


@pytest.mark.asyncio
async def test_list_hot_non_positive_top_n_returns_empty_without_redis_read() -> None:
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zrevrange = AsyncMock()
    tracker = HotTracker(redis, None, Settings(somni_hot_top_n=0))

    assert await tracker.list_hot(language="zh") == []
    redis.zrevrange.assert_not_called()


@pytest.mark.asyncio
async def test_list_hot_requires_redis() -> None:
    from app.core.exceptions import AppError
    from app.server.somni.audio.hot import HotTracker

    tracker = HotTracker(None, None, Settings())
    with pytest.raises(AppError):
        await tracker.list_hot(language="zh")


@pytest.mark.asyncio
async def test_list_hot_redis_failure_is_service_unavailable() -> None:
    from app.core.exceptions import AppError
    from app.server.somni.audio.hot import HotTracker

    redis = MagicMock()
    redis.zrevrange = AsyncMock(side_effect=ConnectionError("unavailable"))
    tracker = HotTracker(redis, None, Settings())

    with pytest.raises(AppError) as exc:
        await tracker.list_hot(language="zh")

    assert exc.value.status_code == 503
