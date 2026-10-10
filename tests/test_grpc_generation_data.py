from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock

import grpc
import pytest
from google.protobuf.json_format import ParseDict

from app.core.config import Settings
from app.server.somni.facts.models import canonical_hash
from app.server.somni.facts.rpc import GenerationDataRpc
from app.server.somni.facts.service import GenerationDataService
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2_grpc as rpc


@asynccontextmanager
async def channel(service, authorizer=None):
    server = grpc.aio.server()
    rpc.add_GenerationDataServiceServicer_to_server(GenerationDataRpc(service, authorizer), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as connection:
            yield rpc.GenerationDataServiceStub(connection)
    finally:
        await server.stop(0)


async def allow_synthetic(context, uid):
    return uid == "u"


def make_service():
    store = AsyncMock()
    store.load_active_plan.return_value = []
    store.load_permissions.return_value = {
        "status": "active",
        "privacy_switches": {
            "profiles": True,
            "plans": True,
            "health": True,
            "observations": True,
            "questionnaires": True,
        },
    }
    store.load_profile.return_value = (None, None)
    store.list_window_sessions.return_value = []
    store.list_execution_events.return_value = {"operations": []}
    for name in (
        "load_questionnaires",
        "list_observations",
        "list_sleep_episodes",
        "load_daily_metrics",
        "load_sleep_debt_rows",
    ):
        getattr(store, name).return_value = []
    return GenerationDataService(None, Settings(_env_file=None), store), store


async def test_default_deny_and_uid_authorization() -> None:
    svc, store = make_service()
    request = pb.GetPlanInputsReq(uid="u", request_id="r", profile_snapshot_id="s")
    async with channel(svc) as client:
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetPlanInputs(request)
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
    store.load_permissions.assert_not_called()
    request.uid = "other"
    async with channel(svc, allow_synthetic) as client:
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetPlanInputs(request)
        assert caught.value.code() == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.parametrize(
    "reminder,target,offset",
    [
        ("2025-12-31", "2026-01-01", "+08:00"),
        ("2024-02-28", "2024-02-29", "-07:00"),
        ("2024-02-29", "2024-03-01", "+05:30"),
    ],
)
async def test_bedtime_roundtrip_dates_and_default_deny(reminder, target, offset):
    from tests.test_somni_sleep_debt import night

    svc, store = make_service()
    store.load_sleep_debt_rows.return_value = [night(reminder, 450, _id="synthetic")]
    request = pb.GetBedtimeReminderInputsReq(
        uid="u", request_id="r", reminder_local_date=reminder, timezone_offset=offset
    )
    async with channel(svc) as client:
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetBedtimeReminderInputs(request)
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
    store.load_permissions.assert_not_called()
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetBedtimeReminderInputs(request)
        assert response.target_local_date == target
        assert response.data.weather.status == pb.unsupported
        assert response.data.sleep_debt.value.HasField("daily_debt_min")
        assert response.data.sleep_debt.value.daily_debt_min == 0
        assert not response.data.sleep_debt.value.HasField("baseline_mean_min")
        assert response.data.sleep_debt.value.source_refs == ["wearable_daily:synthetic"]
        assert response.source_manifest[0].revision == "1"
        assert (
            response.target_window.end_at_utc.seconds - response.target_window.start_at_utc.seconds
            == 86400
        )
        svc.settings.generation_response_bytes = 1
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetBedtimeReminderInputs(request)
        assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


async def test_plan_round_trip_hash_and_missing_id() -> None:
    svc, store = make_service()
    content = {
        "uid": "u",
        "schema_version": "2",
        "record_date": "2026-09-20",
        "status": "active",
        "data_cutoff_at_utc": "2026-09-20T00:00:00Z",
        "generated_at_utc": "2026-09-20T00:00:00Z",
        "long_term_profile": {"age": 30},
        "short_term_profile": {},
        "rolling_baseline": {},
        "profile_quality": {},
        "based_on": {},
        "long_term_content_hash": canonical_hash({"age": 30}),
    }
    request = ParseDict(
        {
            "uid": "u",
            "request_id": "r",
            "inline_profile": {**content, "content_hash": canonical_hash(content)},
        },
        pb.GetPlanInputsReq(),
    )
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetPlanInputs(request)
        assert response.data.profile.status == pb.ok
        assert response.data.previous_blueprint.status == pb.not_found
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetPlanInputs(
                pb.GetPlanInputsReq(uid="u", request_id="r", profile_snapshot_id="missing")
            )
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND
    store.list_sleep_episodes.assert_not_called()


async def test_profile_refresh_answer_id_round_trip_over_wire() -> None:
    svc, store = make_service()
    store.load_questionnaires.return_value = [
        {
            "answer_id": "answer-exact",
            "survey_code": "initial",
            "survey_version": "v1",
            "answers": [],
            "results": [],
        }
    ]
    request = pb.GetProfileInputsReq(
        uid="u",
        request_id="profile-refresh-answer",
        purpose=pb.GetProfileInputsReq.profile_refresh,
        review_end_local_date="2026-10-08",
        timezone_offset="+00:00",
        requested_sections=["questionnaires"],
        answer_id="answer-exact",
    )
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetProfileInputs(request)

    assert response.data.questionnaires.status == pb.ok
    assert response.data.questionnaires.value[0].answer_id == "answer-exact"
    assert store.load_questionnaires.await_args.args[0] == "u"
    assert store.load_questionnaires.await_args.args[2] == "answer-exact"


async def test_feedback_observation_round_trip_over_wire() -> None:
    svc, store = make_service()
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "answer-observation-1",
        "kind": "answer",
        "local_date": "2026-09-23",
        "field_code": "comfort",
        "value": "warm",
    }
    request_fields = pb.GetPlanInputsReq.DESCRIPTOR.fields_by_name
    assert "feedback_record_id" in request_fields
    assert "feedback_source_type" not in request_fields
    request = pb.GetPlanInputsReq(
        uid="u",
        request_id="feedback-request",
        target_local_date="2026-09-23",
        feedback_record_id="answer-observation-1",
    )
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetPlanInputs(request)
    assert response.data.feedback.status == pb.ok
    assert len(response.data.feedback.value) == 1
    assert response.data.feedback.value[0].observation_id == "answer-observation-1"
    assert response.data.feedback.value[0].source_refs == ["observations:answer-observation-1"]
    call_args = store.load_feedback_observation.await_args.args
    assert call_args[:2] == ("u", "answer-observation-1")


async def test_profile_sections_and_presence_over_wire() -> None:
    svc, store = make_service()
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetProfileInputs(
            pb.GetProfileInputsReq(
                uid="u",
                request_id="r",
                purpose=pb.GetProfileInputsReq.initial,
                review_end_local_date="2026-09-21",
                timezone_offset="+08:00",
            )
        )
        assert response.data.facts.status == pb.partial
        assert not response.data.facts.value.HasField("latest_age_days")
        assert response.read_consistency == "best_effort"
    empty = pb.InputNumericStats()
    zero = pb.InputNumericStats(mad=0)
    assert not empty.HasField("mad") and zero.HasField("mad")


async def test_episode_missing_and_response_budget() -> None:
    svc, store = make_service()
    store.find_episode.return_value = None
    async with channel(svc, allow_synthetic) as client:
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetEpisodeEvidence(
                pb.GetEpisodeEvidenceReq(uid="u", episode_id="e", source="s")
            )
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND
        svc.settings.generation_response_bytes = 1
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await client.GetProfileInputs(
                pb.GetProfileInputsReq(
                    uid="u",
                    request_id="r",
                    purpose=pb.GetProfileInputsReq.initial,
                    review_end_local_date="2026-09-21",
                    timezone_offset="+08:00",
                )
            )
        assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


async def test_populated_episode_and_weekly_comparison_round_trip() -> None:
    svc, store = make_service()
    original = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "content_hash": "h1",
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}],
    }
    final = deepcopy(original)
    final.update(
        version=2, snapshot_id="v2", base_plan_version=1, edit_source="manual", source_op_id="edit"
    )
    final["phases"][0]["nodes"][0]["sound"].update(volume=15, nullable=None)
    store.load_plan_revision.side_effect = lambda uid, plan, version, when: (
        original if version == 1 else final
    )
    store.find_episode.return_value = {"episode_id": "e", "source": "s"}
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "g",
        "raw_plan": {"uid": "u"},
        "decision_record": {},
        "profile_ref": {},
        "generated_at": "2026-09-18T00:00:00Z",
        "output_plans": [{"plan_id": "p", "version": 1, "snapshot_id": "v1", "date": "2026-09-18"}],
    }
    store.list_sessions.return_value = [
        {
            "uid": "u",
            "session_id": "session",
            "episode_id": "e",
            "source": "s",
            "plan_id": "p",
            "plan_version": 2,
            "generation_id": "g",
            "started_at": "2026-09-18T00:00:00Z",
            "ended_at": "2026-09-18T03:00:00Z",
            "evidence_complete": True,
        }
    ]
    store.list_execution_events.return_value = {
        "operations": [
            {
                "op_id": "edit",
                "session_id": "session",
                "plan_id": "p",
                "plan_version": 2,
                "origin": "manual",
                "phase": "Soothe",
                "node_key": "main",
                "executed_at": "2026-09-18T01:00:00Z",
                "applied": True,
                "success": True,
                "parameters": {"sound": {"volume": 10}},
            }
        ]
    }
    async with channel(svc, allow_synthetic) as client:
        response = await client.GetEpisodeEvidence(
            pb.GetEpisodeEvidenceReq(uid="u", episode_id="e", source="s")
        )
        diff = response.plan_diffs.value[0]
        assert diff.user_adjusted and diff.user_adjustment_count == 1
        assert diff.original_plan_ref.version == 1 and diff.final_plan_ref.version == 2
        change = next(row for row in diff.diff if row.path.endswith("/volume"))
        assert change.before.number_value == 30 and change.after.number_value == 15
        null_change = next(row for row in diff.diff if row.path.endswith("/nullable"))
        assert not null_change.before_present and null_change.after_present
        assert null_change.after.WhichOneof("kind") == "null_value"
        assert diff.execution_diff[0].after.number_value == 10
        weekly = await client.GetProfileInputs(
            pb.GetProfileInputsReq(
                uid="u",
                request_id="r",
                purpose=pb.GetProfileInputsReq.weekly,
                review_generation_id="g",
                review_end_local_date="2026-09-21",
                timezone_offset="+08:00",
            )
        )
        comparison = weekly.data.plan_comparisons.value[0]
        assert comparison.initial_plan_ref.version == 1
        assert comparison.sessions[0].plan_version == 2
        assert comparison.revisions[0].version == 2
