"""打印查询组装结果。运行：python -m app.server.somni.ask.demo"""

from __future__ import annotations

import json

from app.server.somni.ask.query_builder import (
    QueryBuildError,
    assemble_source_queries,
    describe,
)

USER_ID = "demo-user"
REPORT_ID = "674b1c2e0a1b2c3d4e5f6789"


def render_demo() -> str:
    lines = ["# 各表同时查，归属字段只来自 query", ""]
    lines.extend(_show_plan(_mixed_refs()))
    lines.extend(["", "# 这些条件在进 MongoDB 之前被拒绝", ""])
    for title, refs in _rejected_cases():
        lines.append(f"- {title}: {_reject_message(refs)}")
    return "\n".join(lines)


def _show_plan(refs: list[dict]) -> list[str]:
    result = assemble_source_queries(refs)
    lines = ["同时查: " + ", ".join(item.table_name for item in result.queries)]
    for item in result.queries:
        lines.append(f"find {item.collection} " + _dump(item.mongo_filter))
    return lines


def _mixed_refs() -> list[dict]:
    return [
        {
            "table_name": "somni_sleep_reports",
            "query": {
                "_id": REPORT_ID,
                "uid": USER_ID,
                "record_date": {"$in": ["2026-10-26", "2026-10-27", "2026-10-28"]},
            },
            "filter_key": ["sleep_summary"],
        },
        {
            "table_name": "somni_plans",
            "query": {"date": "2026-10-27", "version": {"$gte": 1, "$lt": 5}},
        },
        {
            "table_name": "somni_wearable_daily_profiles",
            "query": {
                "user_id": USER_ID,
                "record_date": "2026-10-28",
                "generated_at": {"$gte": "2026-10-28T00:00:00+08:00"},
            },
        },
        {
            "table_name": "somni_acoustic_clips",
            "query": {"date": {"$in": ["2026-10-27", "2026-10-28"]}},
        },
    ]


def _rejected_cases() -> list[tuple[str, list[dict]]]:
    return [
        ("$where", [{"table_name": "somni_plans", "query": {"$where": "1==1"}}]),
        (
            "日期字符串用 $gt",
            [
                {
                    "table_name": "somni_sleep_reports",
                    "query": {"record_date": {"$gt": "2026-10-01"}},
                }
            ],
        ),
        (
            "status 做范围",
            [
                {
                    "table_name": "somni_wearable_daily_profiles",
                    "query": {"status": {"$gt": 0}},
                }
            ],
        ),
        (
            "点路径字段",
            [
                {
                    "table_name": "somni_sleep_reports",
                    "query": {"sleep_summary.total_minutes": 480},
                }
            ],
        ),
        (
            "对象里塞运算符",
            [
                {
                    "table_name": "somni_sleep_reports",
                    "query": {"sleep_summary": {"$gt": 1}},
                }
            ],
        ),
        ("未登记的集合", [{"table_name": "somni_users", "query": {"uid": USER_ID}}]),
        (
            "数值字段传入数字字符串",
            [{"table_name": "somni_plans", "query": {"version": "2"}}],
        ),
    ]


def _reject_message(refs: list[dict]) -> str:
    try:
        assemble_source_queries(refs)
    except QueryBuildError as exc:
        return exc.message
    return "未被拒绝"


def _dump(value: object) -> str:
    return json.dumps(describe(value), ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    print(render_demo())


if __name__ == "__main__":
    main()
