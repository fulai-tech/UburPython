"""Profile-only provenance projection and initial data boundary regression."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from google.protobuf.json_format import ParseDict

from app.core.config import Settings
from app.server.somni.facts.models import DataError, canonical_hash
from app.server.somni.facts.profile_provenance import compact_profile_provenance
from app.server.somni.facts.service import GenerationDataService
from app.server.somni.facts.store import FactsStore
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb


def make_service():
    store = AsyncMock()
    store.load_profile.return_value = (None, None)
    for method in (
        "load_questionnaires",
        "list_sleep_episodes",
        "load_daily_metrics",
        "list_daily_sleep_reports",
        "load_sleep_debt_rows",
    ):
        getattr(store, method).return_value = []
    settings = Settings(_env_file=None, generation_cursor_secret="synthetic-provenance-test" * 2)
    return GenerationDataService(None, settings, store), store


async def test_initial_default_never_reads_operations_even_if_they_exist():
    svc, store = make_service()
    store.load_preference_sources.return_value = {"operations": [{"_id": i} for i in range(9000)]}
    result = await svc.get_profile_inputs(
        "u", "r", purpose="initial", review_end_local_date="2026-10-23", timezone_offset="+08:00"
    )
    assert set(result["data"]) == {"prior_profile", "questionnaires", "facts", "sleep_debt"}
    store.load_preference_sources.assert_not_called()
    store.list_sessions.assert_not_called()
    store.load_observations.assert_not_called()
    assert ParseDict(result, pb.GetProfileInputsRes()).ByteSize() < 10000


@pytest.mark.parametrize(
    "section", ["preference_evidence", "execution", "observations", "plan_comparisons"]
)
async def test_initial_rejects_operation_sections_before_io(section):
    svc, store = make_service()
    with pytest.raises(DataError, match="initial_section_not_allowed"):
        await svc.get_profile_inputs(
            "u",
            "r",
            purpose="initial",
            review_end_local_date="2026-10-23",
            timezone_offset="+08:00",
            requested_sections=[section],
        )
    assert store.mock_calls == []


async def test_sensor_projection_never_loads_point_arrays_or_upload_ids():
    store = FactsStore(None, Settings(_env_file=None))
    store._read = AsyncMock(return_value=[])
    await store.list_sleep_episodes("u", "2026-10-01", "2026-10-23", datetime.now(UTC))
    await store.load_daily_metrics("u", "2026-10-01", "2026-10-23", datetime.now(UTC))
    await store.list_daily_sleep_reports("u", "2026-10-01", "2026-10-23", datetime.now(UTC))
    for call in store._read.call_args_list:
        fields = call.args[2].split()
        assert not {
            "raw_data",
            "sleep",
            "physiology",
            "activity",
            "metric_windows",
            "source_upload_record_ids",
            "overnight_vitals",
        } & set(fields)
    assert "raw_data.bed_time" in store._read.call_args_list[1].args[2]


def legacy_profile():
    refs = [{"source_type": "operations", "record_id": str(i)} for i in range(4628)]
    return {
        "uid": "u",
        "schema_version": "0.0.1",
        "profile_snapshot_id": "ps",
        "record_date": "2026-10-22",
        "generated_at": "2026-10-22T00:00:00Z",
        "status": "active",
        "snapshot_revision": 1,
        "long_term_content_hash": "linked",
        "based_on": {"source_refs": refs, "middleware_context": {"source_manifest": refs}},
        "short_term_profile": {"sleep_debt": {"source_manifest": refs, "daily_debt_min": 20}},
        "rolling_baseline": {},
        "profile_quality": {},
    }


async def test_legacy_profile_response_fits_budget_and_retains_hash_link():
    svc, store = make_service()
    short = legacy_profile()
    long = {
        "uid": "u",
        "schema_version": "0.0.1",
        "status": "valid",
        "content_hash": "linked",
        "long_term_profile": {"risk_factors": ["allergy"]},
    }
    store.load_profile.return_value = (short, long)
    original = deepcopy(short)
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="plan_generation",
        review_end_local_date="2026-10-23",
        timezone_offset="+08:00",
        requested_sections=["prior_profile"],
    )
    view = result["data"]["prior_profile"]["value"]
    assert short == original
    assert view["long_term_content_hash"] == "linked"
    assert view["long_term_profile"] == long["long_term_profile"]
    assert view["content_hash"] == canonical_hash(
        {k: v for k, v in view.items() if k != "content_hash"}
    )
    assert (
        ParseDict(result, pb.GetProfileInputsRes()).ByteSize()
        < svc.settings.generation_response_bytes
    )
    assert view["based_on"]["provenance_summary"]["retained_reference_count"] <= 256
    assert len(json.dumps(view["based_on"]).encode()) < 128 * 1024


@pytest.mark.parametrize("fault", ["uid", "hash"])
async def test_compaction_cannot_hide_invalid_profile_link(fault):
    svc, store = make_service()
    short = legacy_profile()
    long = {
        "uid": "other" if fault == "uid" else "u",
        "schema_version": "0.0.1",
        "status": "valid",
        "content_hash": "wrong" if fault == "hash" else "linked",
        "long_term_profile": {},
    }
    store.load_profile.return_value = (short, long)
    with pytest.raises(DataError, match="profile_link_mismatch"):
        await svc._profile("u", "", datetime.now(UTC), {})


async def test_unused_operations_never_enter_manifest():
    svc, store = make_service()
    store.load_preference_sources.return_value = {
        "operations": [{"uid": "u", "_id": str(i), "method": "sound.set"} for i in range(4628)],
        "snapshots": [],
        "sessions": [],
    }
    manifest = {}
    result = await svc._preference_evidence(
        "u", "2026-10-22", "+08:00", datetime.now(UTC), manifest
    )
    assert result["value"]["source_summary"]["input_record_count"] == 4628
    assert manifest == {}


def test_projection_is_stable_and_does_not_mutate_business_content():
    original = legacy_profile()
    projected = compact_profile_provenance(original)
    assert compact_profile_provenance(projected) == projected
    assert original["short_term_profile"]["sleep_debt"]["daily_debt_min"] == 20
    assert len(original["based_on"]["source_refs"]) == 4628


async def test_large_business_content_still_hits_rpc_budget():
    import grpc

    from app.server.somni.facts.rpc import GenerationDataRpc

    svc, store = make_service()
    short = legacy_profile()
    long = {
        "uid": "u",
        "schema_version": "0.0.1",
        "status": "valid",
        "content_hash": "linked",
        "long_term_profile": {"business_value": "x" * 2100000},
    }
    store.load_profile.return_value = (short, long)
    context = AsyncMock()
    context.abort.side_effect = RuntimeError("aborted")
    request = pb.GetProfileInputsReq(
        uid="u",
        request_id="r",
        purpose=pb.GetProfileInputsReq.initial,
        review_end_local_date="2026-10-23",
        timezone_offset="+08:00",
        requested_sections=["prior_profile"],
    )
    with pytest.raises(RuntimeError, match="aborted"):
        await GenerationDataRpc(svc).GetProfileInputs(request, context)
    context.abort.assert_awaited_once_with(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "response_budget_exceeded"
    )
    assert len(long["long_term_profile"]["business_value"]) == 2100000
