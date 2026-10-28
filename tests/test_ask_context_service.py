"""GetContext 读取、投影和并查。"""

from __future__ import annotations

import asyncio
import json
import time

import grpc
import pytest
from bson import ObjectId
from loguru import logger

from app.core.config import Settings
from app.server.somni.ask.errors import AskContextError
from app.server.somni.ask.rpc import AskContextRpc
from app.server.somni.ask.service import AskContextService
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb


def _query(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)

USER_ID = "demo-user"
REPORT_ID = "674b1c2e0a1b2c3d4e5f6789"


class FakeStore:
    def __init__(
        self,
        docs: dict[str, list[dict]] | None = None,
        messages: list[dict] | None = None,
    ):
        self.docs = docs or {}
        self.messages = messages or []
        self.calls: list[tuple] = []
        self.delay = 0.0

    async def find(self, logical: str, mongo_filter: dict) -> list[dict]:
        self.calls.append(("find", logical, mongo_filter))
        if self.delay:
            await asyncio.sleep(self.delay)
        return [dict(doc) for doc in self.docs.get(logical, [])]

    async def find_chat_messages(self, conversation_id: str) -> list[dict]:
        self.calls.append(("chat", conversation_id))
        return [dict(message) for message in self.messages]


def _service(store: FakeStore) -> AskContextService:
    service = AskContextService(None, Settings(_env_file=None))
    service._store = store
    return service


@pytest.mark.asyncio
async def test_session_history_returns_empty_turns() -> None:
    result = await _service(FakeStore()).get_context(
        {
            "context_type": "session_history",
            "query_id": "summary-1",
            "session_id": "ask-session-1",
        }
    )
    assert result == {
        "query_id": "summary-1",
        "session_id": "ask-session-1",
        "turns": [],
    }


@pytest.mark.asyncio
async def test_session_history_pairs_user_and_assistant() -> None:
    store = FakeStore(
        messages=[
            {"message_index": 0, "role": "system", "content": {"type": "session_start"}},
            {"message_index": 1, "role": "assistant", "content": "你好"},
            {"message_index": 2, "role": "user", "content": "昨晚睡得怎么样"},
            {"message_index": 3, "role": "assistant", "content": "睡眠效率是 85%。"},
        ]
    )
    result = await _service(store).get_context(
        {
            "context_type": "session_history",
            "query_id": "summary-2",
            "session_id": "sc-1787910432154-d434657e",
        }
    )
    assert result["turns"] == [
        {"question_id": 2, "text_query": "昨晚睡得怎么样", "answer": "睡眠效率是 85%。"}
    ]
    assert store.calls == [("chat", "sc-1787910432154-d434657e")]


@pytest.mark.asyncio
async def test_session_history_rejects_business_refs() -> None:
    with pytest.raises(AskContextError, match="不能带业务引用") as caught:
        await _service(FakeStore()).get_context(
            {
                "context_type": "session_history",
                "query_id": "summary-1",
                "session_id": "ask-session-1",
                "source_refs": [{"table_name": "somni_plans", "query": _query({"plan_id": "p1"})}],
            }
        )
    assert caught.value.status == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_shapes_zero_one_and_many_records() -> None:
    store = FakeStore(
        {
            "somni_plans": [
                {
                    "_id": ObjectId("674b1c2e0a1b2c3d4e5f6789"),
                    "name": "晚间",
                    "public_url": "secret",
                },
            ],
            "somni_sleep_reports": [
                {"_id": "r1", "record_date": "2026-10-26", "sleep_summary": {"body_battery": 80}},
                {"_id": "r2", "record_date": "2026-10-27", "sleep_summary": {"body_battery": 70}},
            ],
        }
    )
    result = await _service(store).get_context(
        {
            "context_type": "source_data",
            "query_id": "ask-1",
            "source_refs": [
                {
                    "table_name": "somni_plans",
                    "query": _query({"plan_id": "plan-1"}),
                    "filter_key": [],
                },
                {
                    "table_name": "somni_sleep_reports",
                    "query": _query({"record_date": {"$in": ["2026-10-26", "2026-10-27"]}}),
                    "filter_key": ["sleep_summary.body_battery"],
                },
                {"table_name": "somni_acoustic_clips", "query": _query({"tag": "snore"})},
            ],
        }
    )
    plans = json.loads(result["table_data"]["somni_plans"])
    assert isinstance(result["table_data"]["somni_plans"], str)
    assert plans["_id"] == REPORT_ID
    assert "public_url" not in plans
    assert json.loads(result["table_data"]["somni_sleep_reports"]) == [
        {"sleep_summary": {"body_battery": 80}},
        {"sleep_summary": {"body_battery": 70}},
    ]
    assert result["table_data"]["somni_acoustic_clips"] == "[]"
    clip_call = next(
        call for call in store.calls if call[0] == "find" and call[1] == "somni_acoustic_clips"
    )
    assert clip_call[2] == {"tag": "snore"}


@pytest.mark.asyncio
async def test_merges_same_record_and_reports_missing_field() -> None:
    doc = {
        "_id": "plan-1",
        "name": "晚间",
        "phases": [{"phase": "Unwind", "nodes": [{"name": "入睡"}]}],
    }
    store = FakeStore({"somni_plans": [doc]})
    result = await _service(store).get_context(
        {
            "context_type": "source_data",
            "query_id": "ask-2",
            "source_refs": [
                {
                    "table_name": "somni_plans",
                    "query": _query({"plan_id": "plan-1"}),
                    "filter_key": ["name"],
                },
                {
                    "table_name": "somni_plans",
                    "query": _query({"name": "晚间"}),
                    "filter_key": ["phases.phase", "phases.nodes.name"],
                },
            ],
        }
    )
    assert json.loads(result["table_data"]["somni_plans"]) == {
        "name": "晚间",
        "phases": [{"phase": "Unwind", "nodes": [{"name": "入睡"}]}],
    }
    with pytest.raises(AskContextError, match="记录缺少字段") as caught:
        await _service(store).get_context(
            {
                "context_type": "source_data",
                "query_id": "ask-3",
                "source_refs": [
                    {
                        "table_name": "somni_plans",
                        "query": _query({"plan_id": "plan-1"}),
                        "filter_key": ["phases.missing"],
                    }
                ],
            }
        )
    assert caught.value.status == "NOT_FOUND"


@pytest.mark.asyncio
async def test_source_queries_run_together() -> None:
    store = FakeStore(docs={"somni_plans": []})
    store.delay = 0.05
    service = _service(store)
    started = time.perf_counter()
    await service.get_context(
        {
            "context_type": "source_data",
            "query_id": "ask-4",
            "source_refs": [
                {"table_name": "somni_plans", "query": _query({"plan_id": "plan-1"})},
                {"table_name": "somni_acoustic_clips", "query": _query({"tag": "snore"})},
            ],
        }
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 0.09
    assert {call[1] for call in store.calls} == {"somni_plans", "somni_acoustic_clips"}


@pytest.mark.asyncio
async def test_direct_sn_skips_device_lookup_and_keeps_status_valid() -> None:
    store = FakeStore()
    await _service(store).get_context(
        {
            "context_type": "source_data",
            "query_id": "ask-5",
            "source_refs": [
                {"table_name": "somni_acoustic_clips", "query": _query({"sn": "SN-OTHER"})},
                {
                    "table_name": "somni_wearable_daily_profiles",
                    "query": _query({"status": 1.0, "record_date": "2026-10-28"}),
                },
            ],
        }
    )
    wearable = next(call for call in store.calls if call[1] == "somni_wearable_daily_profiles")
    assert wearable[2]["status"] == 1
    assert "user_id" not in wearable[2]
    clip = next(call for call in store.calls if call[1] == "somni_acoustic_clips")
    assert clip[2] == {"sn": "SN-OTHER"}


@pytest.mark.asyncio
async def test_large_result_is_returned() -> None:
    store = FakeStore(
        {"somni_plans": [{"_id": "plan-1", "name": "x" * 70000}]}
    )
    result = await _service(store).get_context(
        {
            "context_type": "source_data",
            "query_id": "ask-6",
            "source_refs": [
                {"table_name": "somni_plans", "query": _query({"plan_id": "plan-1"})}
            ],
        }
    )
    assert json.loads(result["table_data"]["somni_plans"])["name"] == "x" * 70000


@pytest.mark.asyncio
async def test_rpc_returns_turns_for_session_history() -> None:
    request = pb.GetContextReq(
        context_type="session_history",
        query_id="summary-1",
        session_id="ask-session-1",
    )

    class Context:
        async def abort(self, code, message):
            raise AssertionError((code, message))

    response = await AskContextRpc(_service(FakeStore())).GetContext(request, Context())
    assert list(response.turns) == []
    assert response.session_id == "ask-session-1"
    assert response.query_id == "summary-1"


@pytest.mark.asyncio
async def test_failure_log_includes_reason() -> None:
    request = pb.GetContextReq(
        context_type="source_data",
        query_id="ask-missing",
        source_refs=[
            pb.AskContextSourceRef(
                table_name="somni_plans",
                query=_query({"plan_id": "plan-1"}),
                filter_key=["phases.missing"],
            )
        ],
    )
    store = FakeStore(
        {"somni_plans": [{"_id": "plan-1", "phases": [{"phase": "Unwind"}]}]}
    )

    class Context:
        async def abort(self, code, message):
            raise grpc.aio.AbortError()

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)))
    try:
        with pytest.raises(grpc.aio.AbortError):
            await AskContextRpc(_service(store)).GetContext(request, Context())
    finally:
        logger.remove(sink)
    assert any(
        "| ERROR" in line
        and "status=NOT_FOUND" in line
        and "reason=记录缺少字段 phases.missing" in line
        for line in lines
    )
