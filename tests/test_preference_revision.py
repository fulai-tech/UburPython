"""Revision failures must increase reads, never silently reuse stale evidence."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import OperationFailure

from app.core.config import Settings
from app.server.somni.facts.preference_revision import PreferenceRevision
from app.server.somni.facts.service import GenerationDataService


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["insert", "update", "replace", "delete"])
async def test_revision_detects_every_mutation_and_ambiguous_owner(operation):
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    event = {
        "operationType": operation,
        "fullDocument": {} if operation == "delete" else {"uid": "u"},
    }
    stream.try_next = AsyncMock(side_effect=[None, event, None, None])
    db = MagicMock()
    db.watch.return_value = stream
    client = MagicMock()
    client.__getitem__.return_value = db
    revision = PreferenceRevision(client, Settings())
    a = await revision.revision("u")
    b = await revision.revision("u")
    assert a != b
    assert await revision.revision("u") == b


@pytest.mark.asyncio
async def test_stream_failure_disables_shortcut():
    client = MagicMock()
    client.__getitem__.return_value.watch.side_effect = OperationFailure("not a replica set")
    revision = PreferenceRevision(client, Settings())
    assert await revision.revision("u") is None
    assert await revision.revision("u") is None


@pytest.mark.asyncio
async def test_unchanged_memory_skips_sources_including_next_day():
    store = AsyncMock()
    service = GenerationDataService(None, Settings(), store=store)
    service.preference_revision.revision = AsyncMock(return_value="epoch:1")
    store.load_preference_sources.return_value = {"operations": [], "snapshots": [], "sessions": []}
    cold = await service._preference_evidence(
        "u", "2026-10-26", "+08:00", datetime(2026, 10, 26, tzinfo=UTC), {}, profile={}
    )
    profile = {
        "value": {
            "long_term_profile": {
                "learned_preferences": {
                    "policy_version": "somni_preference_memory.v3",
                    "source_revision": cold["value"]["source_revision"],
                }
            }
        }
    }
    store.load_preference_sources.reset_mock()
    for day in ["2026-10-26", "2026-10-27"]:
        result = await service._preference_evidence(
            "u", day, "+08:00", datetime(2026, 10, 27, tzinfo=UTC), {}, profile=profile
        )
        assert result["value"]["unchanged"] is True
    store.load_preference_sources.assert_not_awaited()


@pytest.mark.asyncio
async def test_change_during_source_read_never_publishes_a_trusted_cursor():
    from app.server.somni.facts.models import DataError

    store = AsyncMock()
    store.load_preference_sources.return_value = {"operations": [], "snapshots": [], "sessions": []}
    service = GenerationDataService(None, Settings(), store=store)
    service.preference_revision.revision = AsyncMock(side_effect=["epoch:1", "epoch:2"])
    with pytest.raises(DataError, match="changed_during_read"):
        await service._preference_evidence(
            "u", "2026-10-26", "+08:00", datetime(2026, 10, 26, tzinfo=UTC), {}, profile={}
        )


@pytest.mark.asyncio
async def test_changed_user_does_not_reuse_previous_source_cache():
    store = AsyncMock()
    store.load_preference_sources.return_value = {"operations": [], "snapshots": [], "sessions": []}
    service = GenerationDataService(None, Settings(), store=store)
    service.preference_revision.revision = AsyncMock(side_effect=["a", "a", "b", "b"])
    for _ in range(2):
        await service._preference_evidence(
            "u", "2026-10-26", "+08:00", datetime(2026, 10, 26, tzinfo=UTC), {}, profile={}
        )
    assert store.load_preference_sources.await_count == 2


@pytest.mark.asyncio
async def test_fingerprint_is_stable_across_workers_with_different_stream_epochs():
    values = []
    for cursor in ["worker-a:0:0", "worker-b:4:2"]:
        store = AsyncMock()
        store.load_preference_sources.return_value = {
            "operations": [],
            "snapshots": [],
            "sessions": [],
        }
        service = GenerationDataService(None, Settings(), store=store)
        service.preference_revision.revision = AsyncMock(return_value=cursor)
        result = await service._preference_evidence(
            "u", "2026-10-26", "+08:00", datetime(2026, 10, 26, tzinfo=UTC), {}, profile={}
        )
        values.append(result["value"]["source_revision"])
    assert values[0] == values[1]
