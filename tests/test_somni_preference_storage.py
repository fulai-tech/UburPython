"""Production-shaped Mongo records must work without invented persistence fields."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from app.server.somni.facts.preference import aggregate_preference_evidence


def stored_sources():
    phases = [
        {
            "phase": "Soothe",
            "start_time": "22:00",
            "duration_sec": 1800,
            "nodes": [
                {"duration_sec": 1800, "sound": {"tracks": [{"audio_id": "a", "volume": 30}]}}
            ],
        }
    ]
    before = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "date": "2026-10-15",
        "status": 0,
        "edit_source": "dph",
        "create_time": datetime(2026, 10, 15, 1, tzinfo=UTC),
        "phases": phases,
        "device_id": "d",
    }
    after = deepcopy(before)
    after.update(
        version=2,
        snapshot_id="v2",
        edit_source="app",
        create_time=datetime(2026, 10, 15, 2, tzinfo=UTC),
    )
    after["phases"][0]["nodes"][0]["sound"]["tracks"][0]["volume"] = 20
    session = {
        "session_id": "s",
        "device_id": "d",
        "plan_id": "p",
        "plan_version": 2,
        "started_at": datetime(2026, 10, 15, 14, tzinfo=UTC),
        "ended_at": datetime(2026, 10, 15, 14, 30, tzinfo=UTC),
        "state": "ended",
        "end_reason": "completed",
        "previewing": False,
        "total_nodes": 1,
        "timezone": "Asia/Shanghai",
    }
    return [before, after], [session]


def aggregate(snapshots, sessions, operations=None):
    return aggregate_preference_evidence(
        "u",
        "2026-10-16",
        "+08:00",
        operations or [],
        snapshots,
        sessions,
        read_at=datetime(2026, 10, 16, 1, tzinfo=UTC),
    )


def test_real_storage_fields_prove_adopted_revision_without_persisted_uid():
    snapshots, sessions = stored_sources()
    result = aggregate(snapshots, sessions)
    assert len(result["exposures"]) == 1
    assert result["observations"][0]["after"] == 20
    assert result["observations"][0]["applied"] is True
    assert "uid" not in sessions[0]  # normalization never mutates source


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "running"},
        {"end_reason": "start_failed"},
        {"end_reason": "user_stop"},
        {"previewing": True},
        {"timezone": ""},
        {"timezone": "America/New_York"},
        {"total_nodes": 2},
        {"device_id": "foreign"},
        {"plan_version": 99},
        {"uid": "foreign"},
        {"ended_at": datetime(2026, 10, 15, 14, 10, tzinfo=UTC)},
    ],
)
def test_incomplete_foreign_or_preview_sessions_never_count(changes):
    snapshots, sessions = stored_sources()
    sessions[0].update(changes)
    assert aggregate(snapshots, sessions)["exposures"] == []


@pytest.mark.parametrize(
    "changes",
    [{"apply_status": "discarded"}, {"uid": "foreign"}, {"version": 3}, {"date": "2026-10-14"}],
)
def test_discarded_noncontiguous_or_foreign_revision_never_votes(changes):
    snapshots, sessions = stored_sources()
    snapshots[1].update(changes)
    assert not any(x["applied"] for x in aggregate(snapshots, sessions)["observations"])


def test_duplicate_revision_and_bad_order_are_not_parent_proof():
    snapshots, sessions = stored_sources()
    assert aggregate(snapshots + [deepcopy(snapshots[0])], sessions)["observations"] == []
    snapshots[0]["create_time"] = snapshots[1]["create_time"] + timedelta(seconds=1)
    assert aggregate(snapshots, sessions)["observations"] == []


def test_hardware_real_shape_is_burst_observation_without_hold_vote():
    snapshots, sessions = stored_sources()
    rows = [
        {
            "uid": "u",
            "device_id": "d",
            "op_id": str(i),
            "origin": "local",
            "method": "sound.set",
            "success": True,
            "session_id": "",
            "executed_at": sessions[0]["started_at"] + timedelta(minutes=2, seconds=i),
            "payload": {"type": "sound", "value": 20 + i},
            "applied": {"value": 20 + i},
        }
        for i in range(4)
    ]
    result = aggregate(snapshots, sessions, rows + deepcopy(rows))
    assert len(result["operation_bursts"]) == 1
    burst = result["operation_bursts"][0]
    assert (burst["first_value"], burst["last_value"], burst["step_count"]) == (20, 23, 4)
    hardware = [x for x in result["observations"] if x["source"] == "operation"]
    assert len(hardware) == 1 and hardware[0]["hold_verified"] is False


@pytest.mark.asyncio
async def test_stored_target_requires_exact_revision_and_equal_current_body():
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.models import DataError
    from app.server.somni.facts.service import GenerationDataService

    snapshots, _ = stored_sources()
    pointer = {
        key: deepcopy(snapshots[0][key])
        for key in ("uid", "plan_id", "version", "date", "phases", "device_id")
    }
    store = AsyncMock()
    store.load_active_plan.return_value = [pointer]
    store.load_plan_revision.return_value = snapshots[0]
    service = GenerationDataService(None, Settings(), store=store)
    result = await service._target_plan("u", "2026-10-15", datetime.now(UTC), {})
    assert result["value"]["content_hash"]
    assert "generation_id" not in result["value"]
    pointer["phases"][0]["nodes"][0]["sound"]["tracks"][0]["volume"] = 99
    with pytest.raises(DataError, match="active_plan_body_mismatch"):
        await service._target_plan("u", "2026-10-15", datetime.now(UTC), {})


@pytest.mark.asyncio
async def test_store_reads_owned_revisions_before_uidless_sessions():
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.models import DataError
    from app.server.somni.facts.store import FactsStore

    store = FactsStore(None, Settings())
    store._read = AsyncMock(side_effect=[[], [{"uid": "u", "plan_id": "p"}], [], []])
    await store.load_preference_sources("u", "2026-10-16", "+08:00")
    calls = store._read.call_args_list
    assert calls[1].args[0] == "snapshots"
    assert calls[1].args[1]["$or"][1]["date"]["$lt"] == "2026-10-16"
    assert calls[-1].args[0] == "sessions"
    assert calls[-1].args[1]["plan_id"] == {"$in": ["p"]}
    store._read = AsyncMock(
        side_effect=[[], [{"uid": "u", "plan_id": "p"}], [{"uid": "other", "plan_id": "p"}]]
    )
    with pytest.raises(DataError, match="preference_plan_owner_ambiguous"):
        await store.load_preference_sources("u", "2026-10-16", "+08:00")


def test_utc_midnight_session_is_attributed_to_plan_business_night():
    snapshots, sessions = stored_sources()
    for row in snapshots:
        row["phases"][0]["start_time"] = "23:50"
    sessions[0].update(
        started_at=datetime(2026, 10, 15, 15, 50, tzinfo=UTC),
        ended_at=datetime(2026, 10, 15, 16, 20, tzinfo=UTC),
    )
    result = aggregate(snapshots, sessions)
    assert result["observations"][0]["night"] == "2026-10-15"
    snapshots[1]["create_time"] = datetime(2026, 10, 16, tzinfo=UTC)
    assert aggregate(snapshots, sessions)["exposures"] == []


def test_rejected_snapshot_cannot_create_hardware_observation():
    snapshots, sessions = stored_sources()
    snapshots[1]["apply_status"] = "device_rejected"
    assert aggregate(snapshots, sessions)["exposures"] == []


@pytest.mark.asyncio
async def test_optional_nulls_in_current_plan_do_not_break_stored_comparison():
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.service import GenerationDataService

    snapshots, _ = stored_sources()
    current = deepcopy(snapshots[0])
    current["phases"][0]["nodes"][0]["light"] = {"curve": None}
    snapshots[0]["phases"][0]["nodes"][0]["light"] = {}
    store = AsyncMock()
    store.load_plan_revision.return_value = snapshots[0]
    result = await GenerationDataService(None, Settings(), store=store)._stored_target_plan(
        "u", "2026-10-15", current, datetime.now(UTC), {}
    )
    assert result["status"] == "ok"


def test_changed_node_identity_does_not_become_preference():
    snapshots, sessions = stored_sources()
    snapshots[0]["phases"][0]["nodes"][0]["node_id"] = "old-node"
    snapshots[1]["phases"][0]["nodes"][0]["node_id"] = "new-node"
    assert aggregate(snapshots, sessions)["observations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user", "device", None])
async def test_daily_target_reads_latest_owned_plan_regardless_of_legacy_scope(scope):
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.service import GenerationDataService

    snapshots, _ = stored_sources()
    pointer = {
        key: deepcopy(snapshots[0][key])
        for key in ("uid", "plan_id", "version", "date", "phases", "device_id")
    }
    if scope is not None:
        pointer["scope"] = scope
    store = AsyncMock()
    store.load_latest_user_plan.return_value = [pointer]
    store.load_plan_revision.return_value = snapshots[0]
    read_at = datetime.now(UTC)
    result = await GenerationDataService(None, Settings(), store=store)._target_plan(
        "u", "2026-10-15", read_at, {}, stored_compat=True
    )
    assert result["status"] == "ok"
    assert result["value"]["plan_id"] == "p"
    store.load_latest_user_plan.assert_awaited_once_with("u", "2026-10-15", read_at)
    store.load_active_plan.assert_not_awaited()
    store.load_stored_current_plan.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_missing_plan_does_not_fall_back_to_another_query():
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.service import GenerationDataService

    store = AsyncMock()
    store.load_latest_user_plan.return_value = []
    store.load_active_plan.return_value = [{"uid": "u", "scope": "device"}]
    result = await GenerationDataService(None, Settings(), store=store)._target_plan(
        "u", "2026-10-15", datetime.now(UTC), {}, stored_compat=True
    )
    assert result["status"] == "not_found"
    store.load_active_plan.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid,reason",
    [
        ("owner", "active_plan_owner_mismatch"),
        ("date", "active_plan_date_mismatch"),
        ("version", "active_plan_snapshot_version_mismatch"),
        ("snapshot_id", "active_plan_link_incomplete"),
        ("body", "active_plan_body_mismatch"),
    ],
)
async def test_daily_device_scope_keeps_target_identity_and_revision_checks(invalid, reason):
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.server.somni.facts.models import DataError
    from app.server.somni.facts.service import GenerationDataService

    snapshots, _ = stored_sources()
    pointer = {
        key: deepcopy(snapshots[0][key])
        for key in ("uid", "plan_id", "version", "date", "phases", "device_id")
    }
    pointer["scope"] = "device"
    if invalid == "owner":
        pointer["uid"] = "other"
    elif invalid == "date":
        pointer["date"] = "2026-10-16"
    elif invalid == "version":
        pointer["version"] = 2
    elif invalid == "snapshot_id":
        snapshots[0]["snapshot_id"] = ""
    elif invalid == "body":
        pointer["phases"][0]["nodes"][0]["sound"]["tracks"][0]["volume"] = 99
    store = AsyncMock()
    store.load_latest_user_plan.return_value = [pointer]
    store.load_plan_revision.return_value = snapshots[0]
    with pytest.raises(DataError, match=reason):
        await GenerationDataService(None, Settings(), store=store)._target_plan(
            "u", "2026-10-15", datetime.now(UTC), {}, stored_compat=True
        )
    store.load_latest_user_plan.assert_awaited_once()
    store.load_active_plan.assert_not_awaited()
