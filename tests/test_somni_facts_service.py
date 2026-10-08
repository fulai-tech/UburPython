from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.models import DataError, canonical_hash, timestamp, utc_now
from app.server.somni.facts.service import GenerationDataService


def profile() -> dict:
    value = {
        "uid": "u",
        "schema_version": "2",
        "record_date": "2026-09-20",
        "status": "active",
        "data_cutoff_at_utc": "2026-09-20T00:00:00Z",
        "generated_at_utc": "2026-09-20T00:00:00Z",
        "long_term_profile": {},
        "short_term_profile": {},
        "rolling_baseline": {},
        "profile_quality": {},
        "based_on": {},
        "long_term_content_hash": canonical_hash({}),
    }
    return {**value, "content_hash": canonical_hash(value)}


def service() -> tuple:
    store = AsyncMock()
    store.load_active_plan.return_value = []
    store.load_permissions.return_value = {
        "status": "active",
        "privacy_switches": {
            "profiles": True,
            "plans": True,
            "health": True,
            "questionnaires": True,
            "observations": True,
        },
    }
    settings = Settings(_env_file=None, generation_cursor_secret="synthetic-test-key-" * 3)
    return GenerationDataService(None, settings, store), store


async def test_plan_inline_no_health_queries() -> None:
    svc, store = service()
    result = await svc.get_plan_inputs("u", "req", inline_profile=profile())
    assert result["data"]["profile"]["status"] == "ok"
    assert result["data"]["previous_blueprint"]["status"] == "not_found"
    assert [call[0] for call in store.mock_calls] == ["load_permissions"]
    wrong = profile()
    wrong["uid"] = "other"
    with pytest.raises(DataError, match="owner_mismatch"):
        await svc.get_plan_inputs("u", "req", inline_profile=wrong)


async def test_previous_blueprint_is_loaded_by_exact_generation_id() -> None:
    svc, store = service()
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "weekly-generation-1",
        "status": "committed",
        "raw_plan": {"uid": "u", "revision": 1},
        "decision_record": {"decision_id": "decision-1"},
        "output_plans": [],
    }
    result = await svc.get_plan_inputs(
        "u",
        "blueprint-request",
        inline_profile=profile(),
        previous_generation_id="weekly-generation-1",
    )
    blueprint = result["data"]["previous_blueprint"]
    assert blueprint["value"]["generation_id"] == "weekly-generation-1"
    store.load_blueprint.assert_awaited_once()


async def test_target_plan_ids_are_resolved_from_active_pointer() -> None:
    svc, store = service()
    store.load_active_plan.return_value = [
        {
            "uid": "u",
            "date": "2026-09-23",
            "plan_id": "plan-from-middleware",
            "version": 2,
            "weekly_generation_id": "week-from-middleware",
            "generation_id": "daily-generation",
        }
    ]
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "week-from-middleware",
        "status": "committed",
        "raw_plan": {"uid": "u"},
        "decision_record": {},
        "output_plans": [{"plan_id": "plan-from-middleware", "date": "2026-09-23"}],
    }
    store.load_plan_revision.return_value = {
        "uid": "u",
        "plan_id": "plan-from-middleware",
        "version": 2,
        "snapshot_id": "snapshot-2",
        "content_hash": "hash-2",
        "status": "active",
        "plan_start_local_date": "2026-09-23",
        "phases": [],
    }
    result = await svc.get_plan_inputs(
        "u", "request", inline_profile=profile(), target_local_date="2026-09-23"
    )
    target = result["data"]["target_plan"]["value"]
    assert target["weekly_generation_id"] == "week-from-middleware"
    assert target["plan_id"] == "plan-from-middleware"
    assert target["version"] == 2
    target_only = await svc.get_plan_inputs(
        "u", "request-target-only", target_local_date="2026-09-23"
    )
    assert target_only["data"]["target_plan"]["value"]["plan_id"] == "plan-from-middleware"
    assert target_only["data"]["profile"]["status"] == "not_found"


async def test_feedback_observation_is_exact_and_date_scoped() -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "obs-1",
        "kind": "answer",
        "field_code": "comfort",
        "value": "warm",
        "local_date": "2026-09-23",
    }
    result = await svc.get_plan_inputs(
        "u",
        "feedback-request",
        target_local_date="2026-09-23",
        feedback_record_id="obs-1",
    )
    feedback = result["data"]["feedback"]
    assert feedback["status"] == "ok"
    assert feedback["value"][0]["observation_id"] == "obs-1"
    call_args = store.load_feedback_observation.await_args.args
    assert call_args[:2] == ("u", "obs-1")
    assert isinstance(call_args[2], datetime)


async def test_feedback_record_date_is_normalized_to_proto_local_date() -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "obs-record-date",
        "kind": "answer",
        "record_date": "2026-09-23",
    }
    result = await svc.get_plan_inputs(
        "u",
        "record-date-request",
        target_local_date="2026-09-23",
        feedback_record_id="obs-record-date",
    )
    assert result["data"]["feedback"]["value"][0]["local_date"] == "2026-09-23"


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (None, "feedback_record_not_found"),
        (
            {"uid": "other", "observation_id": "obs-1", "local_date": "2026-09-23"},
            "feedback_record_owner_mismatch",
        ),
        (
            {"uid": "u", "observation_id": "obs-1", "local_date": "2026-09-22"},
            "feedback_record_date_mismatch",
        ),
    ],
)
async def test_feedback_observation_missing_owner_or_date_fails(row, reason) -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = row
    if reason == "feedback_record_not_found":
        result = await svc.get_plan_inputs(
            "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
        )
        assert result["data"]["feedback"]["status"] == "not_found"
        return
    with pytest.raises(DataError, match=reason):
        await svc.get_plan_inputs(
            "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
        )


async def test_feedback_observation_requires_target_date_and_consent() -> None:
    svc, store = service()
    with pytest.raises(DataError, match="feedback_target_date_required"):
        await svc.get_plan_inputs("u", "r", feedback_record_id="obs-1")
    store.load_permissions.return_value = {
        "status": "active",
        "privacy_switches": {"plans": True, "observations": False},
    }
    result = await svc.get_plan_inputs(
        "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
    )
    assert result["data"]["feedback"]["status"] == "unsupported"
    store.load_feedback_observation.assert_not_awaited()


async def test_target_plan_rejects_ambiguous_active_rows() -> None:
    svc, store = service()
    store.load_active_plan.return_value = [{"uid": "u"}, {"uid": "u"}]
    with pytest.raises(DataError, match="active_plan_ambiguous"):
        await svc.get_plan_inputs(
            "u", "request", inline_profile=profile(), target_local_date="2026-09-23"
        )


async def test_explicit_id_missing_and_consent() -> None:
    svc, store = service()
    store.load_profile.return_value = None, None
    with pytest.raises(DataError, match="profile_not_found"):
        await svc.get_plan_inputs("u", "req", profile_snapshot_id="missing")
    store.reset_mock()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {}}
    result = await svc.get_plan_inputs("u", "req", profile_snapshot_id="missing")
    assert result["data"]["profile"]["status"] == "unsupported"
    store.load_profile.assert_not_called()


def test_cursor_signature_binding_and_expiration() -> None:
    svc, _store = service()
    binding = {"uid": "u", "episode_id": "e", "source": "s", "session_id": ""}
    token = svc._cursor_encode({**binding, "read_at": timestamp(utc_now()), "last": ["p", "s"]})
    assert svc._cursor_decode(token, binding)["last"] == ["p", "s"]
    for bad_token, bad_binding in (
        (token[:-1] + "x", binding),
        (token, {**binding, "uid": "other"}),
    ):
        with pytest.raises(DataError, match="invalid_cursor"):
            svc._cursor_decode(bad_token, bad_binding)
    old = svc._cursor_encode({**binding, "read_at": "2020-01-01T00:00:00Z", "last": ["p", "s"]})
    with pytest.raises(DataError, match="invalid_cursor"):
        svc._cursor_decode(old, binding)


async def test_exact_original_final_diff_and_revert() -> None:
    svc, store = service()
    original = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "content_hash": "h1",
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}],
    }
    final = deepcopy(original)
    final.update(version=3, snapshot_id="v3", base_plan_version=1, edit_source="manual")
    final["phases"][0]["nodes"][0]["sound"]["volume"] = 15
    store.load_plan_revision.side_effect = lambda uid, plan, version, when: (
        original if version == 1 else final
    )
    store.list_execution_events.return_value = {
        "operations": [],
        "downlinks": [],
        "events": [],
        "session_payloads": [],
    }
    generation = {"output_plans": [{"plan_id": "p", "version": 1, "snapshot_id": "v1"}]}
    session = {
        "uid": "u",
        "session_id": "s",
        "episode_id": "e",
        "plan_id": "p",
        "plan_version": 3,
        "ended_at": "2026-09-20T00:00:00Z",
        "evidence_complete": True,
    }
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["user_adjusted"] is True and result["user_adjustment_count"] == 1
    assert result["diff"][0]["before"] == 30 and result["diff"][0]["after"] == 15
    assert result["diff"][0]["origins"] == ["manual"]
    final["phases"] = deepcopy(original["phases"])
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["user_adjusted"] is True and result["diff"] == []


async def test_weekly_requires_generation_before_queries() -> None:
    svc, store = service()
    with pytest.raises(DataError, match="weekly_generation_required"):
        await svc.get_profile_inputs(
            "u",
            "req",
            purpose="weekly",
            review_end_local_date="2026-09-21",
            timezone_offset="+08:00",
        )
    store.load_permissions.assert_not_called()


async def test_plan_consent_does_not_grant_execution_health_access() -> None:
    svc, store = service()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {"plans": True}}
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="weekly",
        review_generation_id="g",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
    )
    assert result["data"]["plan_comparisons"]["status"] == "unsupported"
    store.list_sessions.assert_not_called()
    store.list_window_sessions.assert_not_called()


async def test_pagination_rechecks_permissions_and_keeps_read_marker() -> None:
    svc, store = service()
    store.find_episode.return_value = {"episode_id": "e", "source": "s"}
    store.list_sessions.return_value = [
        {"uid": "u", "session_id": str(index), "plan_id": "p", "episode_id": "e"}
        for index in range(3)
    ]
    first = await svc.get_episode_evidence("u", "e", "s", limit=1)
    second = await svc.get_episode_evidence("u", "e", "s", limit=1, cursor=first["next_cursor"])
    assert first["plan_diffs"]["value"][0]["session_id"] == "0"
    assert second["plan_diffs"]["value"][0]["session_id"] == "1"
    assert first["read_started_at_utc"] == second["read_started_at_utc"]
    assert store.load_permissions.await_count == 2
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {}}
    denied = await svc.get_episode_evidence("u", "e", "s", cursor=second["next_cursor"])
    assert denied["plan_diffs"]["status"] == "unsupported"


async def test_prior_version_live_change_cannot_override_later_saved_version() -> None:
    svc, store = service()
    original = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}],
    }
    final = deepcopy(original)
    final.update(version=2, snapshot_id="v2", base_plan_version=1, edit_source="manual")
    final["phases"][0]["nodes"][0]["sound"]["volume"] = 15
    store.load_plan_revision.side_effect = lambda uid, plan, version, when: (
        original if version == 1 else final
    )
    operation = {
        "op_id": "op1",
        "session_id": "s",
        "plan_id": "p",
        "plan_version": 1,
        "executed_at": "2026-09-20T01:00:00Z",
        "phase": "Soothe",
        "node_key": "main",
        "applied": True,
        "success": True,
        "origin": "manual",
        "parameters": {"sound": {"volume": 20}},
    }
    store.list_execution_events.return_value = {"operations": [operation]}
    session = {
        "uid": "u",
        "session_id": "s",
        "plan_id": "p",
        "plan_version": 1,
        "adopted_versions": [{"version": 2, "applied": True, "applied_at": "2026-09-20T02:00:00Z"}],
    }
    generation = {"output_plans": [{"plan_id": "p", "version": 1, "snapshot_id": "v1"}]}
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["final_plan_ref"]["version"] == 2
    assert result["diff"][0]["after"] == 15
    assert result["execution_diff"] == []
