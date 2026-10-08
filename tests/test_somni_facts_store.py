from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.models import DataError
from app.server.somni.facts.store import FactsStore


class Cursor:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def max_time_ms(self, value):
        return self

    def sort(self, value):
        self.sort_spec = value
        return self

    def limit(self, value):
        return self

    async def to_list(self, length):
        return self.rows[:length]

    def batch_size(self, value):
        return self

    def __aiter__(self):
        self.iterator = iter(self.rows)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self):
        self.closed = True


def make_store(rows: list) -> tuple:
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    collection.with_options.return_value = collection
    collection.find.return_value = Cursor(rows)
    return FactsStore(client, Settings(_env_file=None, generation_query_limit=2)), collection


async def test_permissions_projection_and_budget() -> None:
    store, collection = make_store([])
    assert await store.load_permissions("synthetic") is None
    query, projection = collection.find.call_args.args
    assert query == {"uid": "synthetic"}
    assert set(projection) == {"status", "privacy_switches"}
    collection.find.return_value = Cursor([{}, {}, {}])
    with pytest.raises(DataError, match="source_query_budget"):
        await store.load_daily_metrics("synthetic", "2026-09-01", "2026-09-21", datetime.now(UTC))


async def test_feedback_observation_is_exact_and_user_scoped() -> None:
    store, collection = make_store([{"uid": "u", "observation_id": "obs-1"}])
    read_at = datetime(2026, 9, 23, tzinfo=UTC)
    row = await store.load_feedback_observation("u", "obs-1", read_at)
    assert row == {"uid": "u", "observation_id": "obs-1"}
    query, projection = collection.find.call_args.args
    assert query["uid"] == "u"
    assert query["observation_id"] == "obs-1"
    assert query["status"] == "active"
    assert "observation_id" in projection
    assert "record_date" in projection
    assert "value" in projection


async def test_activity_uses_own_date_and_owner() -> None:
    store, collection = make_store([])
    await store.load_daily_metrics("synthetic", "2026-09-01", "2026-09-21", datetime.now(UTC))
    query, projection = collection.find.call_args.args
    assert query["user_id"] == "synthetic"
    assert "activity_date" in query and "record_date" not in query
    assert "activity" in projection


async def test_active_plan_lookup_is_uid_date_and_read_time_scoped() -> None:
    store, collection = make_store([])
    read_at = datetime(2026, 9, 23, tzinfo=UTC)
    await store.load_active_plan("synthetic", "2026-09-23", read_at)
    query, projection = collection.find.call_args.args
    assert query["uid"] == "synthetic"
    assert query["date"] == "2026-09-23"
    assert query["status"] == "active"
    assert query["$or"] == [
        {"received_at": {"$lte": read_at}},
        {"received_at": {"$exists": False}},
    ]
    assert collection.find.call_args.kwargs == {}
    assert collection.find.return_value.sort_spec == [("version", -1), ("_id", -1)]
    assert set(projection) >= {"plan_id", "version", "weekly_generation_id", "generation_id"}


async def test_exact_profile_hash_join_and_no_latest_long() -> None:
    store, collection = make_store([])
    collection.find.side_effect = [
        Cursor([{"snapshot_id": "s1", "schema_version": "2", "long_term_content_hash": "hash"}]),
        Cursor([]),
    ]
    short, long = await store.load_profile("synthetic", "s1", datetime.now(UTC))
    assert short and long is None
    assert collection.find.call_args_list[0].args[0]["snapshot_id"] == "s1"
    assert collection.find.call_args_list[1].args[0]["content_hash"] == "hash"
    assert collection.find.call_args_list[1].args[0]["uid"] == "synthetic"


async def test_no_events_without_owned_sessions() -> None:
    store, collection = make_store([])
    result = await store.list_execution_events(
        "synthetic", [{"uid": "other", "session_id": "x"}], datetime.now(UTC)
    )
    assert result["operations"] == []
    collection.find.assert_not_called()


async def test_more_than_ten_thousand_rows_and_utc_codec() -> None:
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    collection.with_options.return_value = collection
    collection.find.return_value = Cursor([{"record_date": "2026-09-20"}] * 10001)
    store = FactsStore(client, Settings(_env_file=None))
    rows = await store.load_daily_metrics("u", "2026-09-01", "2026-09-21", datetime.now(UTC))
    assert len(rows) == 10001
    assert collection.with_options.call_args.kwargs["codec_options"].tz_aware is True


async def test_sleep_history_scans_past_28_days_and_closes_cursor():
    from tests.test_somni_sleep_debt import CUTOFF, night

    store, collection = make_store([])
    store._settings.generation_query_limit = 100
    rows = [
        night((date(2026, 9, 21) - timedelta(days=3 * index)).isoformat()) for index in range(1, 20)
    ]
    history = Cursor(rows)
    collection.find.side_effect = [Cursor([night("2026-09-21")]), history]
    result = await store.load_sleep_debt_rows("u", "2026-09-21", CUTOFF)
    assert len(result) == 15 and history.closed
    query, projection = collection.find.call_args.args
    assert query["user_id"] == "u" and query["record_date"] == {"$lt": "2026-09-21"}
    assert "device_algorithm" not in query
    assert {"revision", "selection_policy_version", "timezone_offset"} <= projection.keys()


async def test_sleep_history_budget_does_not_return_fallback():
    from tests.test_somni_sleep_debt import CUTOFF, night

    store, collection = make_store([])
    history = Cursor([night("2026-09-20"), night("2026-09-19")])
    collection.find.side_effect = [Cursor([night("2026-09-21")]), history]
    with pytest.raises(DataError, match="sleep_history_budget_exceeded"):
        await store.load_sleep_debt_rows("u", "2026-09-21", CUTOFF)
    assert history.closed
