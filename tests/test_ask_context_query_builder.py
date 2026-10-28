"""AskContext 查询组装。确认交给 find 的是白名单过滤条件，而不是客户端原对象。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId

from app.server.somni.ask.demo import render_demo
from app.server.somni.ask.query_builder import QueryBuildError, assemble_source_queries

USER_ID = "demo-user"
REPORT_ID = "674b1c2e0a1b2c3d4e5f6789"
TZ8 = timezone(timedelta(hours=8))


def test_demo_shows_parallel_batches_and_rejections() -> None:
    text = render_demo()
    assert "同时查" in text
    assert "$objectId" in text
    assert "$date" in text
    assert "$where" in text
    assert "未被拒绝" not in text


def test_query_keeps_table_owner_field_without_injecting_one() -> None:
    client_query = {
        "_id": REPORT_ID,
        "uid": USER_ID,
        "record_date": {"$in": ["2026-10-26", "2026-10-27"]},
    }
    result = assemble_source_queries(
        [{"table_name": "somni_sleep_reports", "query": client_query, "filter_key": ["x"]}],
    )
    found = result.queries[0].mongo_filter
    assert found is not client_query
    assert found["uid"] == USER_ID
    assert isinstance(found["_id"], ObjectId)
    assert str(found["_id"]) == REPORT_ID
    assert found["record_date"] == {"$in": ["2026-10-26", "2026-10-27"]}
    assert "filter_key" not in found
    client_query["record_date"]["$in"].append("2099-01-01")
    assert found["record_date"]["$in"] == ["2026-10-26", "2026-10-27"]


def test_date_field_becomes_aware_datetime_and_status_defaults_to_valid() -> None:
    result = assemble_source_queries(
        [
            {
                "table_name": "somni_wearable_daily_profiles",
                "query": {"generated_at": {"$gte": "2026-10-28T00:00:00+08:00"}},
            }
        ],
    )
    found = result.queries[0].mongo_filter
    assert "user_id" not in found
    assert found["status"] == 1
    assert found["generated_at"]["$gte"] == datetime(2026, 10, 28, tzinfo=TZ8)


def test_acoustic_without_sn_does_not_add_owner() -> None:
    result = assemble_source_queries(
        [{"table_name": "somni_acoustic_clips", "query": {"tag": "snore"}}],
    )
    assert result.queries[0].mongo_filter == {"tag": "snore"}


def test_struct_integral_float_becomes_int() -> None:
    result = assemble_source_queries(
        [
            {
                "table_name": "somni_wearable_daily_profiles",
                "query": {
                    "status": 1.0,
                    "input_sample_count": 2.0,
                    "aggregation_version": {"$gt": 1.0, "$lt": 3.5},
                },
            }
        ],
    )
    found = result.queries[0].mongo_filter
    assert found["status"] == 1
    assert type(found["status"]) is int
    assert found["input_sample_count"] == 2
    assert found["aggregation_version"] == {"$gt": 1, "$lt": 3.5}


def test_acoustic_sn_is_used_as_given() -> None:
    result = assemble_source_queries(
        [{"table_name": "somni_acoustic_clips", "query": {"sn": "SN-OTHER"}}],
    )
    assert result.queries[0].mongo_filter == {"sn": "SN-OTHER"}


def test_tables_are_queried_together_without_injected_uid() -> None:
    result = assemble_source_queries(
        [
            {"table_name": "somni_plans", "query": {"plan_id": "plan-1"}},
            {"table_name": "somni_acoustic_clips", "query": {"date": "2026-10-28"}},
            {"table_name": "somni_sleep_reports", "query": {"uid": USER_ID}},
        ],
    )
    assert [item.table_name for item in result.queries] == [
        "somni_plans",
        "somni_acoustic_clips",
        "somni_sleep_reports",
    ]
    assert "uid" not in result.queries[0].mongo_filter
    assert result.queries[1].mongo_filter["date"] == "2026-10-28"
    assert result.queries[2].mongo_filter["uid"] == USER_ID


@pytest.mark.parametrize(
    ("refs", "message"),
    [
        ([{"table_name": "somni_plans", "query": {"$where": "1"}}], "不允许的查询字段"),
        (
            [
                {
                    "table_name": "somni_sleep_reports",
                    "query": {"record_date": {"$gt": "2026-10-01"}},
                }
            ],
            "等值或",
        ),
        (
            [{"table_name": "somni_wearable_daily_profiles", "query": {"status": {"$gt": 0}}}],
            "等值",
        ),
        (
            [{"table_name": "somni_wearable_daily_profiles", "query": {"status": 0}}],
            "作废",
        ),
        (
            [{"table_name": "somni_plans", "query": {"version": {"$gt": 1, "$gte": 2}}}],
            "重复",
        ),
        (
            [{"table_name": "somni_plans", "query": {"version": {"$gt": 5, "$lt": 5}}}],
            "范围为空",
        ),
        (
            [{"table_name": "somni_plans", "query": {"version": "2"}}],
            "数值",
        ),
        (
            [
                {
                    "table_name": "somni_sleep_reports",
                    "query": {"create_time": {"$gte": "2026-10-01"}},
                }
            ],
            "不允许查询字段",
        ),
        (
            [
                {
                    "table_name": "somni_wearable_daily_profiles",
                    "query": {"latest_upload_record_id": REPORT_ID},
                }
            ],
            "ObjectId",
        ),
        (
            [{"table_name": "somni_users", "query": {"uid": USER_ID}}],
            "四张表",
        ),
        ([{"table_name": "somni_plans", "query": {}}], "非空"),
        (
            [{"table_name": "somni_sleep_reports", "query": {"sleep_summary": {"$gt": 1}}}],
            "运算符",
        ),
    ],
)
def test_rejects_injection_and_illegal_ranges(refs: list[dict], message: str) -> None:
    with pytest.raises(QueryBuildError, match=message):
        assemble_source_queries(refs)


def test_in_list_is_capped() -> None:
    start = datetime(2024, 1, 1, tzinfo=TZ8)
    days = [(start + timedelta(days=offset)).date().isoformat() for offset in range(366)]
    assemble_source_queries(
        [{"table_name": "somni_plans", "query": {"date": {"$in": days}}}],
    )
    days.append((start + timedelta(days=366)).date().isoformat())
    with pytest.raises(QueryBuildError, match="366"):
        assemble_source_queries(
            [{"table_name": "somni_plans", "query": {"date": {"$in": days}}}],
        )
