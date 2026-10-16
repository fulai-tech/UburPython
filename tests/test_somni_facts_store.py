from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.models import DataError
from app.server.somni.facts.store import FactsStore


class Cursor:
    def __init__(self, rows: list) -> None:
        self.rows = [
            {
                "received_at": datetime(2000, 1, 1, tzinfo=UTC),
                "generated_at": datetime(2000, 1, 1, tzinfo=UTC),
                "created_at": datetime(2000, 1, 1, tzinfo=UTC),
                **row,
            }
            for row in rows
        ]

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


async def test_projection_and_budget() -> None:
    store, collection = make_store([])
    assert await store._read("users", {"uid": "synthetic"}, "status", limit=1) == []
    query, projection = collection.find.call_args.args
    assert query == {"uid": "synthetic"}
    assert set(projection) == {"status"}
    collection.find.return_value = Cursor([{}, {}, {}])
    with pytest.raises(DataError, match="source_query_budget"):
        await store.load_daily_metrics("synthetic", "2026-09-01", "2026-09-21", datetime.now(UTC))


@pytest.mark.parametrize("invalid_received", [None, datetime(2026, 10, 14, tzinfo=UTC)])
async def test_partial_facts_read_keeps_verified_rows_and_reports_excluded_times(
    invalid_received,
) -> None:
    cutoff = datetime(2026, 10, 13, tzinfo=UTC)
    store, _ = make_store(
        [
            {"record_date": "2026-10-11", "received_at": cutoff - timedelta(days=1)},
            {"record_date": "2026-10-12", "received_at": invalid_received},
        ]
    )
    reasons = []
    rows = await store._read(
        "records",
        store._visible({"uid": "synthetic"}, cutoff),
        "record_date",
        missing_reasons=reasons,
    )
    assert [row["record_date"] for row in rows] == ["2026-10-11"]
    reason = (
        "source_time_unverified" if invalid_received is None else "historical_version_unavailable"
    )
    assert reasons == [f"records_{reason}"]
    with pytest.raises(DataError, match=reason):
        await store._read("records", store._visible({"uid": "synthetic"}, cutoff), "record_date")


async def test_feedback_observation_is_exact_and_user_scoped() -> None:
    store, collection = make_store([{"uid": "u", "observation_id": "obs-1"}])
    read_at = datetime(2026, 9, 23, tzinfo=UTC)
    row = await store.load_feedback_observation("u", "obs-1", read_at)
    assert row["uid"] == "u" and row["observation_id"] == "obs-1"
    query, projection = collection.find.call_args.args
    assert query["uid"] == "u"
    assert query["observation_id"] == "obs-1"
    assert query["status"] == "active"
    assert "observation_id" in projection
    assert "record_date" in projection
    assert "value" in projection


async def test_quiz_questions_are_loaded_by_id_without_visibility_filter() -> None:
    question_id = "a0f15f100000000000000001"
    store, collection = make_store(
        [
            {
                "_id": question_id,
                "input_type": "radio",
                "options": {"items": [{"value": "A", "label": "18—24岁"}]},
            }
        ]
    )
    loaded = await store.load_quiz_questions([question_id, question_id])
    query, projection = collection.find.call_args.args
    assert projection == {"input_type": 1, "options": 1}
    assert "received_at" not in query
    assert loaded[question_id]["options"]["items"][0]["label"] == "18—24岁"


async def test_questionnaire_query_filters_the_exact_answer_id() -> None:
    store, collection = make_store(
        [
            {
                "uid": "u",
                "answer_id": "answer-exact",
                "survey_code": "initial",
                "survey_version": "v1",
                "answer_revision": 1,
                "completed_at": datetime(2026, 9, 20, tzinfo=UTC),
                "answers": [],
            }
        ]
    )
    rows = await store.load_questionnaires(
        "u", datetime(2026, 9, 21, tzinfo=UTC), answer_id="answer-exact"
    )
    query = collection.find.call_args_list[0].args[0]
    assert query["uid"] == "u"
    assert query["$or"] == [{"answer_id": "answer-exact"}]
    assert rows[0]["answer_id"] == "answer-exact"


async def test_activity_uses_own_date_and_owner() -> None:
    store, collection = make_store([])
    await store.load_daily_metrics("synthetic", "2026-09-01", "2026-09-21", datetime.now(UTC))
    query, projection = collection.find.call_args.args
    assert query["user_id"] == "synthetic"
    assert "activity_date" in query and "record_date" not in query
    assert "activity" in projection


async def test_daily_sleep_reports_are_user_date_and_cutoff_scoped() -> None:
    store, collection = make_store([])
    read_at = datetime(2026, 10, 8, tzinfo=UTC)

    await store.list_daily_sleep_reports("synthetic", "2026-09-10", "2026-10-08", read_at)

    query, projection = collection.find.call_args.args
    assert query["uid"] == "synthetic"
    assert query["record_date"] == {"$gte": "2026-09-10", "$lt": "2026-10-08"}
    assert query["$and"][0]["$or"] == [
        {"received_at": {"$lte": read_at}},
        {"received_at": {"$not": {"$type": "date"}}},
    ]
    assert set(projection) == {
        "uid",
        "reports_id",
        "record_date",
        "language",
        "overnight_vitals",
        "received_at",
        "create_time",
        "update_time",
        "created_at",
        "generated_at",
        "completed_at",
    }


async def test_questionnaire_query_filters_exact_answer_id() -> None:
    store, collection = make_store(
        [
            {
                "uid": "u",
                "answer_id": "answer-exact",
                "survey_code": "initial",
                "survey_version": "v1",
                "answer_revision": 1,
                "completed_at": datetime(2026, 9, 20, tzinfo=UTC),
                "answers": [],
            }
        ]
    )
    result = await store.load_questionnaires(
        "u", datetime(2026, 9, 21, tzinfo=UTC), answer_id="answer-exact"
    )
    query = collection.find.call_args_list[0].args[0]
    assert query["uid"] == "u"
    assert query["$or"] == [{"answer_id": "answer-exact"}]
    assert result[0]["answer_id"] == "answer-exact"


async def test_active_plan_lookup_is_uid_date_and_read_time_scoped() -> None:
    store, collection = make_store([])
    read_at = datetime(2026, 9, 23, tzinfo=UTC)
    await store.load_active_plan("synthetic", "2026-09-23", read_at)
    query, projection = collection.find.call_args_list[0].args
    assert query["uid"] == "synthetic"
    assert query["date"] == "2026-09-23"
    assert query["status"] == "active"
    assert query["$and"][0]["$or"] == [
        {"received_at": {"$lte": read_at}},
        {"received_at": {"$not": {"$type": "date"}}},
    ]
    assert collection.find.call_args.kwargs == {}
    assert collection.find.return_value.sort_spec == [("version", -1), ("_id", -1)]
    assert set(projection) >= {"plan_id", "version", "weekly_generation_id", "generation_id"}


async def test_previous_active_plan_queries_strictly_before_target_date() -> None:
    store, collection = make_store([{"uid": "u", "date": "2026-10-05"}])
    read_at = datetime(2026, 10, 8, tzinfo=UTC)
    assert await store.load_latest_active_plan_date_before("u", "2026-10-12", read_at) == (
        "2026-10-05"
    )
    query, projection = collection.find.call_args.args
    assert query["uid"] == "u"
    assert query["date"] == {"$lt": "2026-10-12"}
    assert query["status"] == "active"
    assert set(projection) == {
        "uid",
        "date",
        "received_at",
        "create_time",
        "update_time",
        "created_at",
        "generated_at",
        "completed_at",
    }
    assert collection.find.return_value.sort_spec == [("date", -1), ("_id", -1)]


async def test_plan_history_detects_disabled_record_without_active_pointer() -> None:
    store, collection = make_store([{"uid": "u", "date": "2026-10-05", "status": "disabled"}])
    assert await store.has_plan_history_before("u", "2026-10-12", datetime(2026, 10, 8, tzinfo=UTC))
    query, _projection = collection.find.call_args.args
    assert query["status"] == {"$in": ["active", "disabled"]}
    assert query["date"] == {"$lt": "2026-10-12"}


async def test_previous_active_plan_and_history_can_both_be_absent() -> None:
    store, collection = make_store([])
    read_at = datetime(2026, 10, 8, tzinfo=UTC)
    latest = await store.load_latest_active_plan_date_before("u", "2026-10-12", read_at)
    assert latest is None
    assert not await store.has_plan_history_before("u", "2026-10-12", read_at)
    assert collection.find.call_count == 3
    queries = [call.args[0] for call in collection.find.call_args_list]
    assert all(query["uid"] == "u" and query["date"] == {"$lt": "2026-10-12"} for query in queries)
    assert queries[0]["status"] == "active"
    assert "status" not in queries[1]
    assert queries[2]["status"] == {"$in": ["active", "disabled"]}


async def test_exact_profile_hash_join_and_no_latest_long() -> None:
    store, collection = make_store([])
    collection.find.side_effect = [
        Cursor([{"snapshot_id": "s1", "schema_version": "2", "long_term_content_hash": "hash"}]),
        Cursor([]),
    ]
    read_at = datetime.now(UTC)
    short, long = await store.load_profile("synthetic", "s1", read_at)
    assert short and long is None
    conditions = collection.find.call_args_list[0].args[0]["$and"]
    assert conditions[0] == {"$or": [{"profile_snapshot_id": "s1"}, {"snapshot_id": "s1"}]}
    assert conditions[1] == {
        "$or": [
            {"received_at": {"$lte": read_at}},
            {"received_at": {"$not": {"$type": "date"}}},
        ]
    }
    assert collection.find.call_args_list[1].args[0]["content_hash"] == "hash"
    assert collection.find.call_args_list[1].args[0]["uid"] == "synthetic"


async def test_profile_lookup_rejects_ambiguous_ids() -> None:
    store, collection = make_store([{"snapshot_id": "s1"}, {"profile_snapshot_id": "s1"}])
    with pytest.raises(DataError, match="profile_snapshot_ambiguous"):
        await store.load_profile("u", "s1", datetime.now(UTC))
    assert collection.find.call_count == 1




async def test_profile_lookup_selects_latest_when_multiple_snapshots_without_snapshot_id() -> None:
    store, collection = make_store([
        {
            "snapshot_id": "s_new",
            "profile_snapshot_id": "s_new",
            "record_date": "2026-10-15",
            "snapshot_revision": 1,
            "schema_version": "0.0.1",
            "long_term_content_hash": "hash1",
        },
        {
            "snapshot_id": "s_old",
            "profile_snapshot_id": "s_old",
            "record_date": "2026-10-15",
            "snapshot_revision": 1,
            "schema_version": "0.0.1",
            "long_term_content_hash": "hash2",
        },
    ])
    collection.find.side_effect = [
        Cursor([
            {
                "snapshot_id": "s_new",
                "profile_snapshot_id": "s_new",
                "record_date": "2026-10-15",
                "snapshot_revision": 1,
                "schema_version": "0.0.1",
                "long_term_content_hash": "hash1",
            },
            {
                "snapshot_id": "s_old",
                "profile_snapshot_id": "s_old",
                "record_date": "2026-10-15",
                "snapshot_revision": 1,
                "schema_version": "0.0.1",
                "long_term_content_hash": "hash2",
            },
        ]),
        Cursor([
            {
                "uid": "u",
                "schema_version": "0.0.1",
                "content_hash": "hash1",
                "status": "valid",
                "long_term_profile": {"basic_info": {}},
            }
        ]),
    ]
    short, long = await store.load_profile("u", "", datetime.now(UTC), end_date="2026-10-15")
    assert short is not None
    assert short["snapshot_id"] == "s_new"
    assert long is not None

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


@pytest.mark.parametrize("offset", [-1, 0, 1])
async def test_received_time_boundaries(offset: int) -> None:
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    store, _collection = make_store([{"received_at": cutoff + timedelta(seconds=offset)}])
    if offset > 0:
        with pytest.raises(DataError, match="historical_version_unavailable"):
            await store.list_daily_sleep_reports("u", "2026-10-01", "2026-10-11", cutoff)
    else:
        rows = await store.list_daily_sleep_reports("u", "2026-10-01", "2026-10-11", cutoff)
        assert len(rows) == 1


@pytest.mark.parametrize("received", [None, "not-a-time"])
async def test_missing_time_is_not_reported_as_no_history(received) -> None:
    store, _collection = make_store([{"received_at": received}])
    with pytest.raises(DataError, match="source_time_unverified"):
        await store.load_profile("u", "exact-snapshot", datetime(2026, 10, 10, tzinfo=UTC))


async def test_overwritten_version_is_not_returned_as_historical_data() -> None:
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    store, _collection = make_store(
        [
            {
                "received_at": cutoff - timedelta(days=1),
                "update_time": cutoff + timedelta(seconds=1),
            }
        ]
    )
    with pytest.raises(DataError, match="historical_version_unavailable"):
        await store.load_profile("u", "exact-snapshot", cutoff)


def test_visibility_preserves_original_disjunction_and_version_bounds() -> None:
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    original = {
        "uid": "u",
        "$or": [{"revision": 1}, {"revision": 2}],
        "$and": [{"status": "active"}],
    }
    query = FactsStore._visible(original, cutoff)
    assert query["uid"] == "u"
    assert query["$or"] == original["$or"]
    assert query["$and"][0] == {"status": "active"}
    assert {
        "$or": [{"generated_at": {"$lte": cutoff}}, {"generated_at": {"$not": {"$type": "date"}}}]
    } in query["$and"]
    assert original["$and"] == [{"status": "active"}]


async def test_superseded_profile_is_not_false_cold_start() -> None:
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    store, collection = make_store([])
    collection.find.side_effect = [
        Cursor([]),
        Cursor(
            [
                {
                    "uid": "u",
                    "status": "superseded",
                    "received_at": cutoff - timedelta(days=1),
                }
            ]
        ),
    ]
    with pytest.raises(DataError, match="historical_version_unavailable"):
        await store.load_profile("u", "", cutoff, "2026-10-10")
    assert collection.find.call_args_list[0].args[0]["status"] == "active"
    assert "status" not in collection.find.call_args_list[1].args[0]


async def test_saved_profile_without_generation_time_is_unverified() -> None:
    store, _collection = make_store([{"uid": "u", "generated_at": None}])
    with pytest.raises(DataError, match="source_time_unverified"):
        await store.load_profile("u", "", datetime(2026, 10, 10, tzinfo=UTC), "2026-10-10")


@pytest.mark.parametrize("later_update", [False, True])
async def test_database_version_times_replace_missing_receipt(later_update: bool) -> None:
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    updated = cutoff + timedelta(seconds=1) if later_update else cutoff
    store, _collection = make_store(
        [
            {
                "uid": "u",
                "received_at": None,
                "create_time": cutoff - timedelta(days=1),
                "update_time": updated,
            }
        ]
    )
    if later_update:
        with pytest.raises(DataError, match="historical_version_unavailable"):
            await store.load_profile("u", "snapshot", cutoff)
    else:
        short, _long = await store.load_profile("u", "snapshot", cutoff)
        assert short["received_at"] == updated


async def test_database_answer_object_id_and_completion_aliases() -> None:
    from bson import ObjectId

    identifier = ObjectId()
    cutoff = datetime(2026, 10, 10, tzinfo=UTC)
    store, collection = make_store(
        [
            {
                "_id": identifier,
                "survey_code": "initial",
                "received_at": None,
                "create_time": cutoff,
                "update_time": cutoff,
                "answers": [],
            }
        ]
    )
    rows = await store.load_questionnaires("u", cutoff, str(identifier))
    assert rows[0]["answer_id"] == str(identifier)
    assert rows[0]["completed_at"] == cutoff
    query = collection.find.call_args.args[0]
    assert query["uid"] == "u"
    assert {"_id": identifier} in query["$or"]


async def test_current_profile_identifier_alias() -> None:
    store, _collection = make_store([{"profile_snapshot_id": "current", "uid": "u"}])
    short, _long = await store.load_profile("u", "current", datetime(2026, 10, 10, tzinfo=UTC))
    assert short["snapshot_id"] == "current"
