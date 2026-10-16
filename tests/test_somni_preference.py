from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.preference import aggregate_preference_evidence, merge_operation_bursts
from app.server.somni.facts.store import FactsStore


def operation(second: int, value: int) -> dict:
    return {
        "uid": "u",
        "op_id": str(second),
        "sn": "d",
        "session_id": "s",
        "plan_id": "p",
        "plan_version": 2,
        "phase": "Soothe",
        "node_key": "0",
        "resource_id": "a",
        "origin": "local",
        "method": "sound.set",
        "success": True,
        "executed_at": datetime(2026, 10, 15, 15, 0, second, tzinfo=UTC),
        "applied": {"execution_state": "APPLIED", "value": value},
    }


def test_knob_burst_deduplicates_and_retains_endpoints() -> None:
    rows = [operation(0, 40), operation(1, 41), operation(2, 42)]
    result = merge_operation_bursts(rows + deepcopy(rows))
    assert len(result) == 1
    assert (result[0]["first_value"], result[0]["last_value"], result[0]["step_count"]) == (
        40,
        42,
        3,
    )
    assert result[0]["hold_verified"] is False


def test_switch_boundary_and_time_gap_split_bursts() -> None:
    switch = {**operation(1, 40), "method": "control.switch"}
    result = merge_operation_bursts([operation(0, 40), switch, operation(2, 41), operation(5, 42)])
    assert len(result) == 3


def test_unknown_apply_and_failed_commands_are_excluded() -> None:
    first = operation(0, 40)
    first["applied"]["execution_state"] = "QUEUED"
    second = {**operation(1, 41), "success": False}
    assert merge_operation_bursts([first, second]) == []


def sources() -> tuple[list, list]:
    phase = [
        {"phase": "Soothe", "nodes": [{"sound": {"tracks": [{"audio_id": "a", "volume": 30}]}}]}
    ]
    before = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "a",
        "phases": phase,
        "plan_start_local_date": "2026-10-15",
        "timezone_offset": "+08:00",
        "lifecycle": "active",
    }
    after = deepcopy(before)
    after.update(
        version=2,
        snapshot_id="b",
        base_plan_version=1,
        edit_source="manual",
        created_at=datetime(2026, 10, 15, 10, tzinfo=UTC),
    )
    after["phases"][0]["nodes"][0]["sound"]["tracks"][0]["volume"] = 20
    session = {
        "uid": "u",
        "session_id": "s",
        "device_id": "d",
        "plan_id": "p",
        "plan_version": 2,
        "started_at": datetime(2026, 10, 15, 15, tzinfo=UTC),
        "ended_at": datetime(2026, 10, 16, 0, tzinfo=UTC),
        "evidence_complete": True,
        "executed_nodes": [{"phase": "Soothe", "node_key": "0", "completed": True}],
    }
    return [before, after], [session]


def aggregate(snapshots: list, sessions: list, ops: list | None = None) -> dict:
    return aggregate_preference_evidence(
        "u",
        "2026-10-16",
        "+08:00",
        ops or [],
        snapshots,
        sessions,
        read_at=datetime(2026, 10, 16, 1, tzinfo=UTC),
    )


def test_manual_adopted_snapshot_is_one_night_in_local_timezone() -> None:
    snapshots, sessions = sources()
    result = aggregate(snapshots, sessions)
    assert result["observations"][0]["night"] == "2026-10-15"
    assert result["observations"][0]["applied"] is True
    assert len(result["exposures"]) == 1
    assert result["observations"][0]["after"] == 20


def test_snapshot_count_and_metadata_alone_are_not_preferences() -> None:
    snapshots, sessions = sources()
    snapshots[1]["phases"] = deepcopy(snapshots[0]["phases"])
    assert aggregate(snapshots, sessions)["observations"] == []


def test_foreign_uid_missing_coverage_and_ongoing_are_not_automatic_evidence() -> None:
    snapshots, sessions = sources()
    sessions[0]["uid"] = "other"
    assert aggregate(snapshots, sessions)["exposures"] == []
    sessions[0]["uid"] = "u"
    sessions[0].pop("executed_nodes")
    assert aggregate(snapshots, sessions)["exposures"] == []
    sessions[0].pop("ended_at")
    assert aggregate(snapshots, sessions)["deferred"] is True


@pytest.mark.asyncio
async def test_store_uses_business_windows_and_keeps_query_budget() -> None:
    store = FactsStore(None, Settings())
    store._read = AsyncMock(return_value=[])
    await store.load_preference_sources("u", "2026-10-16", "+08:00")
    calls = store._read.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.args[1]["uid"] == "u"
        assert "limit" not in call.kwargs
        assert not {"received_at", "created_at", "update_time"} & call.args[1].keys()
    assert calls[0].args[1]["executed_at"]["$gte"] == datetime(2026, 10, 1, 16, tzinfo=UTC)


def test_ambiguous_same_timestamp_is_not_invented_order() -> None:
    bursts = merge_operation_bursts([operation(0, 40), operation(0, 50)])
    assert bursts[0]["ambiguous_order"] is True


def test_snapshot_invalid_status_is_not_exposure() -> None:
    snapshots, sessions = sources()
    snapshots[1]["lifecycle"] = "deleted"
    assert aggregate(snapshots, sessions)["exposures"] == []


@pytest.mark.asyncio
async def test_profile_preference_section_serializes_without_cutoff() -> None:
    from google.protobuf.json_format import MessageToDict, ParseDict

    from app.server.somni.facts.service import GenerationDataService
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb

    snapshots, sessions = sources()
    store = AsyncMock()
    store.load_preference_sources.return_value = {
        "operations": [],
        "snapshots": snapshots,
        "sessions": sessions,
    }
    service = GenerationDataService(None, Settings(), store=store)
    result = await service.get_profile_inputs(
        "u",
        "request",
        purpose="plan_generation",
        review_end_local_date="2026-10-17",
        timezone_offset="+08:00",
        requested_sections=["preference_evidence"],
    )
    encoded = ParseDict(result, pb.GetProfileInputsRes()).SerializeToString()
    restored = MessageToDict(
        pb.GetProfileInputsRes.FromString(encoded), preserving_proto_field_name=True
    )
    assert restored["data"]["preference_evidence"]["value"]["observations"][0]["after"] == 20
    assert "cutoff" not in str(restored)
    store.load_preference_sources.assert_awaited_once_with("u", "2026-10-16", "+08:00")
