"""用提取的检索参数回放本地 /api/audio/search（单 query）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


def _materials_count(data: object) -> int | None:
    if isinstance(data, dict):
        materials = data.get("materials")
        if isinstance(materials, list):
            return len(materials)
        results = data.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and isinstance(first.get("materials"), list):
                return len(first["materials"])
    if isinstance(data, list):
        return len(data)
    return None


def _to_single_payload(params: object) -> dict:
    """兼容旧日志里包成 queries 的参数，统一打成单 query 请求体。"""
    if not isinstance(params, dict):
        return {}
    queries = params.get("queries")
    if isinstance(queries, list) and queries and isinstance(queries[0], dict):
        return queries[0]
    return params


async def _one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    item: dict,
    url: str,
) -> dict:
    payload = _to_single_payload(item.get("params"))
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:500]}
            code = body.get("code") if isinstance(body, dict) else None
            data = body.get("data") if isinstance(body, dict) else None
            materials_count = _materials_count(data)
            result = {
                "index": idx,
                "timestamp": item.get("timestamp"),
                "http_status": resp.status_code,
                "code": code,
                "msg": body.get("msg") if isinstance(body, dict) else None,
                "elapsed_ms": elapsed_ms,
                "materials_count": materials_count,
                "ok": resp.status_code == 200 and code == 200,
            }
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            result = {
                "index": idx,
                "timestamp": item.get("timestamp"),
                "http_status": None,
                "code": None,
                "msg": str(exc),
                "elapsed_ms": elapsed_ms,
                "materials_count": None,
                "ok": False,
            }
        if (idx + 1) % 20 == 0 or idx == 0 or idx + 1 == total:
            print(
                f"[{idx + 1}/{total}] ok={result['ok']} "
                f"http={result['http_status']} code={result['code']} "
                f"ms={result['elapsed_ms']} materials={result['materials_count']}",
                flush=True,
            )
        return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="logs/2026-07-16_search_params_top200.json",
        help="检索参数 JSON",
    )
    parser.add_argument(
        "--output",
        default="logs/2026-07-16_search_replay_top200_results.json",
        help="回放结果 JSON",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (
        root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    requests_data = payload["requests"]
    if args.limit > 0:
        requests_data = requests_data[: args.limit]

    url = f"{args.base_url.rstrip('/')}/api/audio/search"
    total = len(requests_data)
    print(
        f"replay start: total={total} concurrency={args.concurrency} url={url}",
        flush=True,
    )

    t0 = time.perf_counter()
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency + 2,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        tasks = [
            _one(client, sem, i, total, item, url)
            for i, item in enumerate(requests_data)
        ]
        results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = total - ok_count
    elapsed_list = [r["elapsed_ms"] for r in results]
    summary = {
        "mode": "single",
        "source": str(input_path),
        "url": url,
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "concurrency": args.concurrency,
        "wall_sec": round(elapsed, 2),
        "elapsed_ms_avg": round(sum(elapsed_list) / total, 1) if total else 0,
        "elapsed_ms_p50": sorted(elapsed_list)[total // 2] if total else 0,
        "elapsed_ms_max": max(elapsed_list) if total else 0,
    }

    output = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"written={output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
