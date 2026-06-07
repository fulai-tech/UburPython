#!/usr/bin/env python3
"""检索召回率 + 流水线顺序评测（HTTP 打真实服务，读 uburnode.log 校验步骤日志）。

用法:
  .venv/bin/python scripts/eval_search_recall.py
  .venv/bin/python scripts/eval_search_recall.py --base-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx

from app.core.config import get_settings

STEP_PATTERNS = (
    re.compile(r"检索步骤1/4 睡眠阶段过滤：候选数=(\d+)"),
    re.compile(r"检索步骤2/4 内容形态准入：通过数=(\d+)"),
    re.compile(r"检索步骤3/4 厌恶剔除\+粗排：剩余数=(\d+)"),
    re.compile(r"检索步骤4/4 精排截断：top_k=(\d+)，输出数=(\d+)，match_count序列=\[(.*)\]"),
    re.compile(r"检索：睡眠阶段无匹配，短路返回空结果"),
    re.compile(r"检索：内容形态准入无匹配，短路返回空结果"),
)


@dataclass
class RecallCase:
    name: str
    body: dict
    expect: list[str]
    min_recall: float = 1.0  # 期望最低召回率（0~1）


@dataclass
class OrderProof:
    name: str
    body: dict
    checks: list[str] = field(default_factory=list)
    forbidden_names: list[str] = field(default_factory=list)
    first_name: str | None = None
    match_counts_desc: bool = False


# —— 多标签召回用例 ——
RECALL_CASES: list[RecallCase] = [
    RecallCase(
        "放松·3标签",
        {"sleep_stage_tags": ["放松"], "content_tags": ["雨声", "森林", "白噪音"], "top_k": 20},
        ["【测试】放松·自然四合一", "【测试】放松·森林白噪音", "【测试】放松·白噪雨声"],
        min_recall=1.0,
    ),
    RecallCase(
        "放松·4标签",
        {
            "sleep_stage_tags": ["放松"],
            "content_tags": ["雨声", "森林", "白噪音", "溪流"],
            "top_k": 20,
        },
        ["【测试】放松·自然四合一"],
        min_recall=1.0,
    ),
    RecallCase(
        "入睡·3标签",
        {"sleep_stage_tags": ["入睡"], "content_tags": ["海浪", "雨声", "摇篮曲"], "top_k": 20},
        ["【测试】入睡·海雨助眠", "【测试】入睡·海浪摇篮", "【测试】入睡·四重助眠"],
        min_recall=1.0,
    ),
    RecallCase(
        "入睡·4标签含模糊雨声",
        {
            "sleep_stage_tags": ["入睡"],
            "content_tags": ["海浪", "雨声", "摇篮曲", "哼唱"],
            "top_k": 20,
        },
        ["【测试】入睡·四重助眠", "【测试】入睡·雨声摇篮"],
        min_recall=1.0,
    ),
    RecallCase(
        "守护·3标签",
        {"sleep_stage_tags": ["守护"], "content_tags": ["白噪音", "心跳", "哼唱"], "top_k": 20},
        ["【测试】守护·白噪心跳", "【测试】守护·三重安抚", "【测试】守护·故事哼唱"],
        min_recall=1.0,
    ),
    RecallCase(
        "清醒·3标签",
        {"sleep_stage_tags": ["清醒"], "content_tags": ["鸟鸣", "自然", "日光冥想"], "top_k": 20},
        ["【测试】清醒·晨间鸟鸣", "【测试】清醒·鸟鸣日光", "【测试】清醒·晨间四合一"],
        min_recall=1.0,
    ),
    RecallCase(
        "清醒·4标签",
        {
            "sleep_stage_tags": ["清醒"],
            "content_tags": ["鸟鸣", "自然", "日光冥想", "播客"],
            "top_k": 20,
        },
        ["【测试】清醒·晨间四合一"],
        min_recall=1.0,
    ),
    RecallCase(
        "放松·5标签粗排榜首",
        {
            "sleep_stage_tags": ["放松"],
            "content_tags": ["雨声", "森林", "白噪音", "溪流", "篝火"],
            "top_k": 5,
        },
        ["【测试】放松·自然四合一"],
        min_recall=1.0,
    ),
]

# —— 流水线顺序行为证明 ——
ORDER_PROOFS: list[OrderProof] = [
    OrderProof(
        "步骤1短路：未传睡眠阶段",
        {"sleep_stage_tags": [], "content_tags": ["雨声"], "top_k": 10},
        checks=["step1_zero", "short_sleep"],
    ),
    OrderProof(
        "步骤1先于步骤2：错阶段有内容也不召回",
        {"sleep_stage_tags": ["清醒"], "content_tags": ["雨声"], "top_k": 10},
        forbidden_names=["【测试】放松·深夜雨声", "【测试】入睡·下雨的声音"],
        checks=["step1_ran"],
    ),
    OrderProof(
        "步骤2短路：对阶段无内容标签",
        {"sleep_stage_tags": ["放松"], "content_tags": ["完全不存在的标签xyz"], "top_k": 10},
        checks=["step1_positive", "step2_zero", "short_content"],
    ),
    OrderProof(
        "步骤3厌恶剔除在准入之后",
        {
            "sleep_stage_tags": ["清醒"],
            "content_tags": ["咖啡店", "嘈杂"],
            "disliked_tags": ["嘈杂"],
            "top_k": 10,
        },
        forbidden_names=["【测试】清醒·嘈杂咖啡"],
        checks=["step3_filtered"],
    ),
    OrderProof(
        "步骤4粗排：match_count 降序",
        {
            "sleep_stage_tags": ["放松"],
            "content_tags": ["雨声", "森林", "白噪音", "溪流"],
            "top_k": 10,
        },
        first_name="【测试】放松·自然四合一",
        match_counts_desc=True,
        checks=["step4_rank"],
    ),
]


def _log_size(log_path: Path) -> int:
    return log_path.stat().st_size if log_path.exists() else 0


def _read_log_since(log_path: Path, offset: int) -> str:
    if not log_path.exists():
        return ""
    with log_path.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def _parse_last_pipeline(log_chunk: str) -> dict[str, object]:
    lines = [
        ln
        for ln in log_chunk.splitlines()
        if "检索步骤" in ln or "检索：睡眠" in ln or "检索：内容" in ln
    ]
    block = lines if lines else []
    info: dict[str, object] = {}
    for ln in block:
        if m := STEP_PATTERNS[0].search(ln):
            info["step1"] = int(m.group(1))
        elif m := STEP_PATTERNS[1].search(ln):
            info["step2"] = int(m.group(1))
        elif m := STEP_PATTERNS[2].search(ln):
            info["step3"] = int(m.group(1))
        elif m := STEP_PATTERNS[3].search(ln):
            seq_raw = m.group(3).strip()
            info["step4_counts"] = (
                [int(x.strip()) for x in seq_raw.split(",") if x.strip()] if seq_raw else []
            )
        elif STEP_PATTERNS[4].search(ln):
            info["short_sleep"] = True
        elif STEP_PATTERNS[5].search(ln):
            info["short_content"] = True
    return info


async def _search(client: httpx.AsyncClient, base: str, body: dict) -> tuple[list[str], list[int]]:
    resp = await client.post(f"{base}/api/audio/search", json=body)
    payload = resp.json()
    if resp.status_code != 200 or payload.get("code") != 200:
        raise RuntimeError(f"search failed: {resp.status_code} {payload}")
    results = payload.get("data", {}).get("audios", [])
    names = [r["audio_name"] for r in results]
    # match_count 不对外暴露，粗排顺序即代理
    return names, list(range(len(names)))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    settings = get_settings()
    log_path = settings.log_dir_path / settings.log_file_name

    total_exp = 0
    total_hit = 0
    recall_rows: list[tuple[str, float, list[str], list[str]]] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        print("=" * 76)
        print("一、多标签召回率评测")
        print("=" * 76)
        for case in RECALL_CASES:
            names, _ = await _search(client, args.base_url, case.body)
            hit = [n for n in case.expect if n in names]
            miss = [n for n in case.expect if n not in names]
            exp_n = len(case.expect)
            hit_n = len(hit)
            recall = hit_n / exp_n if exp_n else 1.0
            total_exp += exp_n
            total_hit += hit_n
            ok = recall >= case.min_recall
            recall_rows.append((case.name, recall, hit, miss))
            print(f"\n[{'OK' if ok else 'MISS'}] {case.name}")
            print(f"  查询标签数: {len(case.body.get('content_tags', []))}")
            print(f"  召回: {hit_n}/{exp_n} = {recall * 100:.1f}%")
            print(f"  命中: {names[:8]}{'...' if len(names) > 8 else ''}")
            if miss:
                print(f"  未召回: {miss}")

        print("\n" + "=" * 76)
        print("二、流水线顺序证明（行为 + 日志）")
        print("=" * 76)
        order_ok = 0
        for proof in ORDER_PROOFS:
            log_offset = _log_size(log_path)
            names, _ = await _search(client, args.base_url, proof.body)
            await asyncio.sleep(0.2)
            pipe = _parse_last_pipeline(_read_log_since(log_path, log_offset))

            failures: list[str] = []
            if "step1_zero" in proof.checks and pipe.get("step1", -1) != 0:
                failures.append(f"步骤1候选应=0，实际={pipe.get('step1')}")
            if "short_sleep" in proof.checks and not pipe.get("short_sleep"):
                failures.append("应出现睡眠阶段短路日志")
            if "step1_ran" in proof.checks and "step1" not in pipe:
                failures.append("日志中未见步骤1")
            if "step1_positive" in proof.checks and int(pipe.get("step1", 0)) <= 0:
                failures.append(f"步骤1候选应>0，实际={pipe.get('step1')}")
            if "step2_zero" in proof.checks and pipe.get("step2", -1) != 0:
                failures.append(f"步骤2通过应=0，实际={pipe.get('step2')}")
            if "short_content" in proof.checks and not pipe.get("short_content"):
                failures.append("应出现内容准入短路日志")
            if "step3_filtered" in proof.checks:
                s2, s3 = int(pipe.get("step2", 0)), int(pipe.get("step3", 0))
                if s2 > 0 and s3 >= s2:
                    failures.append(f"步骤3应剔除候选（step3 < step2），实际 step2={s2} step3={s3}")
            if proof.forbidden_names:
                bad = [n for n in proof.forbidden_names if n in names]
                if bad:
                    failures.append(f"不应命中: {bad}")
            if proof.first_name and names and names[0] != proof.first_name:
                failures.append(f"榜首应为 {proof.first_name}，实际 {names[0] if names else '空'}")
            if proof.match_counts_desc:
                counts = pipe.get("step4_counts", [])
                if counts and counts != sorted(counts, reverse=True):
                    failures.append(f"match_count 未降序: {counts}")

            ok = not failures
            order_ok += int(ok)
            print(f"\n[{'OK' if ok else 'FAIL'}] {proof.name}")
            print(f"  请求: {proof.body}")
            print(f"  结果数: {len(names)}")
            print(f"  日志流水线: {pipe}")
            if failures:
                print(f"  失败原因: {failures}")

    overall = total_hit / total_exp * 100 if total_exp else 0
    print("\n" + "=" * 76)
    print(f"多标签总体召回率: {total_hit}/{total_exp} = {overall:.1f}%")
    print(f"流水线顺序证明: {order_ok}/{len(ORDER_PROOFS)} 通过")
    print("=" * 76)
    print("\n流水线固定顺序（代码 retrieval.py search 方法）:")
    print("  1. filter_by_sleep_stage → 无候选短路")
    print("  2. _apply_content_admission → 无通过短路")
    print("  3. _apply_dislike_and_coarse_rank")
    print("  4. sorted(match_count)[:top_k]")


if __name__ == "__main__":
    asyncio.run(main())
