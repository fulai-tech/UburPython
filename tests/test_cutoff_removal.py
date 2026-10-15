"""Current-data profile/plan reads no longer expose visibility cutoffs."""

from datetime import UTC, datetime

from test_somni_sleep_debt import night

from app.server.somni.facts.sleep_debt import calculate_sleep_debt
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as messages


def test_proto_retires_cutoff_fields():
    for cls in (
        messages.GetProfileInputsReq,
        messages.GetPlanInputsReq,
        messages.GenerationProfile,
        messages.SleepDebt,
    ):
        assert "data_cutoff_at_utc" not in cls.DESCRIPTOR.fields_by_name


def test_sleep_debt_accepts_late_received_current_version():
    row = night("2026-09-21", 420, received_at="2099-01-01T00:00:00Z")
    result = calculate_sleep_debt([row], "2026-09-21")
    assert result["value"]["daily_debt_min"] == 30
    assert "data_cutoff_at_utc" not in result["value"]


async def test_profile_read_keeps_late_versions_without_changing_other_reads():
    from test_somni_facts_store import make_store

    from app.server.somni.facts.service import GenerationDataService
    from app.server.somni.facts.time import current_generation_read

    future = datetime(2099, 1, 1, tzinfo=UTC)
    store, collection = make_store(
        [
            {
                "_id": "answer-1",
                "answer_id": "answer-1",
                "survey_code": "initial",
                "survey_version": "1",
                "completed_at": datetime(2026, 9, 20, tzinfo=UTC),
                "received_at": future,
                "update_time": future,
                "answers": [],
            }
        ]
    )
    svc = GenerationDataService(None, store._settings, store)
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="initial",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
        requested_sections=["questionnaires"],
        data_cutoff_at_utc="malformed legacy field is ignored",
    )
    assert result["data"]["questionnaires"]["value"][0]["answer_id"] == "answer-1"
    assert result["source_manifest"][0]["received_at_utc"].startswith("2099")
    query = collection.find.call_args.args[0]
    assert "received_at" not in repr(query) and "update_time" not in repr(query)
    assert not current_generation_read()
    assert "received_at" in repr(store._visible({"uid": "u"}, datetime.now(UTC)))


async def test_generation_scope_is_reset_after_error_and_is_task_local():
    import asyncio

    from app.server.somni.facts.time import current_generation_inputs, current_generation_read

    entered, release = asyncio.Event(), asyncio.Event()

    @current_generation_inputs
    async def read(*, fail=False):
        assert current_generation_read()
        entered.set()
        await release.wait()
        if fail:
            raise RuntimeError("injected read failure")
        return {"data_cutoff_at": "old", "nested": {"cutoff_policy_version": "old"}}

    task = asyncio.create_task(read())
    await entered.wait()
    assert not current_generation_read()
    release.set()
    assert await task == {"nested": {}}
    import pytest

    with pytest.raises(RuntimeError, match="injected read failure"):
        await read(fail=True)
    assert not current_generation_read()


async def test_current_session_window_preserves_business_dates_without_read_time_cap():
    from test_somni_facts_store import make_store

    from app.server.somni.facts.time import current_generation_inputs

    store, collection = make_store([])
    read_at = datetime(2026, 9, 21, tzinfo=UTC)

    @current_generation_inputs
    async def read():
        await store.list_window_sessions("u", "2026-09-01", "2026-09-22", read_at, read_at, read_at)

    await read()
    query = collection.find.call_args.args[0]
    assert "started_at" not in query
    assert "sleep_date_local" in repr(query) and "2026-09-22" in repr(query)


async def test_current_read_preserves_time_normalization_and_projection():
    from test_somni_facts_store import make_store

    from app.server.somni.facts.time import current_generation_inputs

    updated = datetime(2099, 1, 1, tzinfo=UTC)
    store, collection = make_store([{"received_at": None, "update_time": updated}])

    @current_generation_inputs
    async def read():
        return await store._read("records", {"uid": "u"}, "record_date received_at")

    rows = await read()
    assert rows[0]["received_at"] == updated
    projection = collection.find.call_args.args[1]
    assert "update_time" in projection and "create_time" in projection
