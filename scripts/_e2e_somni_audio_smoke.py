"""一次性量产音频 gRPC 冒烟：只用带标记的测试数据，结束时全部删除。

用法（服务已启动）：
  .venv/bin/python scripts/_e2e_somni_audio_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpc
from google.protobuf.json_format import MessageToDict

from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

MARKER = f"__e2e_cursor_{uuid.uuid4().hex[:10]}__"
GRPC_TARGET = "127.0.0.1:50064"
# 自定义标签仅测 CRUD；检索/处方必须用已审阅英文词典 code
TAG_CODE = f"e2e_cursor_{uuid.uuid4().hex[:8]}"
CONTENT_CODE = "natural_sound"
AUDIO_NAME_ZH = f"{MARKER}测试雨声"
AUDIO_NAME_EN = f"{MARKER}Test Rain"
AUDIO_URL_ZH = f"https://cdn.example.invalid/e2e/{MARKER}/zh.mp3"
AUDIO_URL_EN = f"https://cdn.example.invalid/e2e/{MARKER}/en.mp3"


def _ok(name: str, detail: str = "") -> None:
    print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))


async def _refresh_es() -> None:
    from app.core.config import get_settings
    from app.es.client import create_es_client

    settings = get_settings()
    client = create_es_client(settings, node=settings.effective_somni_es_node)
    try:
        await client.indices.refresh(
            index=",".join(
                [
                    settings.somni_es_audio_index,
                    settings.somni_es_material_locales_index,
                    settings.somni_es_tag_vectors_index,
                ]
            )
        )
    finally:
        await client.close()


async def run() -> None:
    print(f"标记={MARKER}")
    print(f"连接 gRPC {GRPC_TARGET}")
    created_material_ids: list[str] = []
    created_tag_ids: list[str] = []
    results: list[tuple[str, bool, str]] = []
    material_id = ""

    async with grpc.aio.insecure_channel(GRPC_TARGET) as channel:
        stub = uburnode_somni_pb2_grpc.AudioServiceStub(channel)

        async def step(name: str, coro) -> bool:
            try:
                detail = await coro
                _ok(name, detail or "")
                results.append((name, True, detail or ""))
                return True
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  {name} — {exc}")
                results.append((name, False, str(exc)))
                return False

        try:
            async def create_tags() -> str:
                res = await stub.CreateAudioTag(
                    uburnode_somni_pb2.CreateAudioTagReq(
                        items=[
                            uburnode_somni_pb2.AudioTagInput(
                                type="content_form",
                                code=TAG_CODE,
                                language="zh",
                                name=f"{MARKER}稳定雨",
                                status="启用",
                            ),
                            uburnode_somni_pb2.AudioTagInput(
                                type="content_form",
                                code=TAG_CODE,
                                language="en",
                                name=f"{MARKER}Steady Rain",
                                status="启用",
                            ),
                        ]
                    )
                )
                for item in res.items:
                    if not item.ok:
                        raise RuntimeError(f"创建标签失败: {item.message}")
                    created_tag_ids.append(item.id)
                return f"tags={created_tag_ids}"

            await step("CreateAudioTag zh+en", create_tags())

            async def create_material_zh() -> str:
                res = await stub.CreateAudio(
                    uburnode_somni_pb2.CreateAudioReq(
                        items=[
                            uburnode_somni_pb2.AudioMaterialInput(
                                audio_name=AUDIO_NAME_ZH,
                                language="zh",
                                audio_url=AUDIO_URL_ZH,
                                cover_url=f"https://cdn.example.invalid/e2e/{MARKER}/zh.png",
                                description=f"{MARKER}助眠雨声描述",
                                summary=f"{MARKER}短描述",
                                status=True,
                                source="e2e_test",
                                sleep_stage_tags=uburnode_somni_pb2.AudioTagRefList(
                                    items=[
                                        uburnode_somni_pb2.AudioTagRef(
                                            code="unwind", name="放松"
                                        )
                                    ]
                                ),
                                content_form_tags=uburnode_somni_pb2.AudioContentTagList(
                                    items=[
                                        uburnode_somni_pb2.AudioContentTag(
                                            code=CONTENT_CODE, name="自然声"
                                        )
                                    ]
                                ),
                            )
                        ]
                    )
                )
                item = res.items[0]
                if not item.ok:
                    raise RuntimeError(item.message)
                created_material_ids.append(item.id)
                return f"id={item.id}"

            if await step("CreateAudio zh", create_material_zh()):
                material_id = created_material_ids[0]

            if material_id:

                async def add_en_locale() -> str:
                    res = await stub.UpdateAudio(
                        uburnode_somni_pb2.UpdateAudioReq(
                            items=[
                                uburnode_somni_pb2.UpdateAudioItem(
                                    id=material_id,
                                    fields=uburnode_somni_pb2.AudioMaterialInput(
                                        language="en",
                                        audio_name=AUDIO_NAME_EN,
                                        audio_url=AUDIO_URL_EN,
                                        cover_url=f"https://cdn.example.invalid/e2e/{MARKER}/en.png",
                                        description=f"{MARKER} rain description",
                                        summary=f"{MARKER} short",
                                    ),
                                )
                            ]
                        )
                    )
                    item = res.items[0]
                    if not item.ok:
                        raise RuntimeError(item.message)
                    return "en locale upserted"

                await step("UpdateAudio add en locale", add_en_locale())
                await _refresh_es()
                await asyncio.sleep(0.8)

                async def get_audio_zh() -> str:
                    res = await stub.GetAudio(
                        uburnode_somni_pb2.GetAudioReq(
                            page=1,
                            page_size=20,
                            query_text=MARKER,
                            language="zh",
                        )
                    )
                    ids = [item.id for item in res.list]
                    if material_id not in ids:
                        raise RuntimeError(f"未搜到测试原料，got={ids[:5]} total={res.total}")
                    hit = next(item for item in res.list if item.id == material_id)
                    if MARKER not in hit.audio_name:
                        raise RuntimeError(f"中文名不对: {hit.audio_name}")
                    if hit.audio_url != AUDIO_URL_ZH:
                        raise RuntimeError(f"中文 url 不对: {hit.audio_url}")
                    return f"hit name={hit.audio_name} url={hit.audio_url}"

                await step("GetAudio search zh", get_audio_zh())

                async def get_audio_en() -> str:
                    res = await stub.GetAudio(
                        uburnode_somni_pb2.GetAudioReq(
                            page=1,
                            page_size=20,
                            query_text=AUDIO_NAME_EN,
                            language="en",
                        )
                    )
                    ids = [item.id for item in res.list]
                    if material_id not in ids:
                        raise RuntimeError(f"英文搜索未命中，ids={ids[:5]} total={res.total}")
                    hit = next(item for item in res.list if item.id == material_id)
                    data = MessageToDict(hit, preserving_proto_field_name=True)
                    return (
                        "hit "
                        f"audio_name={data.get('audio_name')!r} "
                        f"audio_name_en={data.get('audio_name_en')!r} "
                        f"audio_url={data.get('audio_url')!r}"
                    )

                await step("GetAudio search en", get_audio_en())

                async def get_by_ids() -> str:
                    res = await stub.GetAudioByIds(
                        uburnode_somni_pb2.GetAudioByIdsReq(
                            ids=[material_id], language="zh"
                        )
                    )
                    if len(res.list) != 1:
                        raise RuntimeError(f"期望1条，got={len(res.list)}")
                    item = res.list[0]
                    if item.audio_url != AUDIO_URL_ZH:
                        raise RuntimeError(f"url={item.audio_url}")
                    if item.audio_name != AUDIO_NAME_ZH:
                        raise RuntimeError(f"name={item.audio_name}")
                    return f"name={item.audio_name}"

                await step("GetAudioByIds zh", get_by_ids())

                async def get_by_ids_en() -> str:
                    res = await stub.GetAudioByIds(
                        uburnode_somni_pb2.GetAudioByIdsReq(
                            ids=[material_id], language="en"
                        )
                    )
                    if len(res.list) != 1:
                        raise RuntimeError(f"期望1条，got={len(res.list)}")
                    item = res.list[0]
                    if item.audio_url != AUDIO_URL_EN:
                        raise RuntimeError(f"en url={item.audio_url}")
                    if item.audio_name != AUDIO_NAME_EN:
                        raise RuntimeError(f"en name={item.audio_name}")
                    return f"name={item.audio_name} url={item.audio_url}"

                await step("GetAudioByIds en", get_by_ids_en())

                async def get_tags() -> str:
                    res = await stub.GetAudioTag(uburnode_somni_pb2.GetAudioTagReq())
                    codes = {t.code for t in res.tags}
                    if TAG_CODE not in codes:
                        raise RuntimeError(f"标签字典未含 {TAG_CODE}")
                    return f"tags_total={len(res.tags)}"

                await step("GetAudioTag", get_tags())

                async def search_audio() -> str:
                    res = await stub.SearchAudio(
                        uburnode_somni_pb2.SearchAudioReq(
                            query_text=MARKER,
                            sleep_stage_tags=["unwind"],
                            content_tag_codes=[CONTENT_CODE],
                            top_k=5,
                        )
                    )
                    ids = []
                    for mat in res.materials:
                        d = MessageToDict(mat, preserving_proto_field_name=True)
                        ids.append(d.get("id") or d.get("_id") or "")
                    if material_id not in ids:
                        raise RuntimeError(f"SearchAudio 未命中 ids={ids}")
                    return f"materials={len(ids)}"

                await step("SearchAudio", search_audio())

                async def prescription_zh() -> str:
                    res = await stub.SearchPrescriptionAudio(
                        uburnode_somni_pb2.SearchPrescriptionAudioReq(
                            tasks=[
                                uburnode_somni_pb2.PrescriptionAudioSearchTask(
                                    task_id="e2e-1",
                                    query_text=MARKER,
                                    sleep_stage_tags=["unwind"],
                                    content_tag_codes=[CONTENT_CODE],
                                    top_k=5,
                                    language="zh",
                                )
                            ]
                        )
                    )
                    if not res.results:
                        raise RuntimeError("无 results")
                    cands = res.results[0].candidates
                    ids = []
                    urls = []
                    for cand in cands:
                        d = MessageToDict(cand, preserving_proto_field_name=True)
                        ids.append(d.get("id") or d.get("_id") or "")
                        urls.append(d.get("audio_url") or "")
                    if material_id not in ids:
                        raise RuntimeError(f"处方未命中测试原料 ids={ids}")
                    url = urls[ids.index(material_id)]
                    if url and url != AUDIO_URL_ZH:
                        raise RuntimeError(f"处方 zh url 期望 {AUDIO_URL_ZH} 实际 {url}")
                    return f"candidates={len(cands)} url={url!r}"

                await step("SearchPrescriptionAudio zh", prescription_zh())

                async def prescription_en() -> str:
                    res = await stub.SearchPrescriptionAudio(
                        uburnode_somni_pb2.SearchPrescriptionAudioReq(
                            tasks=[
                                uburnode_somni_pb2.PrescriptionAudioSearchTask(
                                    task_id="e2e-en",
                                    query_text=AUDIO_NAME_EN,
                                    sleep_stage_tags=["unwind"],
                                    content_tag_codes=[CONTENT_CODE],
                                    top_k=5,
                                    language="en",
                                )
                            ]
                        )
                    )
                    cands = res.results[0].candidates if res.results else []
                    ids = []
                    urls = []
                    for cand in cands:
                        d = MessageToDict(cand, preserving_proto_field_name=True)
                        ids.append(d.get("id") or d.get("_id") or "")
                        urls.append(d.get("audio_url") or "")
                    if material_id not in ids:
                        raise RuntimeError(f"处方 en 未命中 ids={ids}")
                    idx = ids.index(material_id)
                    if urls[idx] != AUDIO_URL_EN:
                        raise RuntimeError(f"处方 en url 期望 {AUDIO_URL_EN} 实际 {urls[idx]}")
                    return f"url={urls[idx]}"

                await step("SearchPrescriptionAudio en", prescription_en())

                async def update_summary() -> str:
                    res = await stub.UpdateAudio(
                        uburnode_somni_pb2.UpdateAudioReq(
                            items=[
                                uburnode_somni_pb2.UpdateAudioItem(
                                    id=material_id,
                                    fields=uburnode_somni_pb2.AudioMaterialInput(
                                        language="zh",
                                        summary=f"{MARKER}短描述已更新",
                                    ),
                                )
                            ]
                        )
                    )
                    if not res.items[0].ok:
                        raise RuntimeError(res.items[0].message)
                    await _refresh_es()
                    detail = await stub.GetAudioByIds(
                        uburnode_somni_pb2.GetAudioByIdsReq(
                            ids=[material_id], language="zh"
                        )
                    )
                    desc = detail.list[0].description
                    if "已更新" not in desc:
                        raise RuntimeError(f"summary 未反映到 description 出参: {desc}")
                    return f"description={desc}"

                await step("UpdateAudio summary", update_summary())

                async def get_hot() -> str:
                    res = await stub.GetHot(
                        uburnode_somni_pb2.GetHotReq(language="zh", kind="query")
                    )
                    return f"items={len(res.items)}"

                await step("GetHot", get_hot())

        finally:
            # 无论成败都清理测试数据
            print("\n--- 清理测试数据 ---")
            try:
                if created_material_ids:
                    res = await stub.DeleteAudio(
                        uburnode_somni_pb2.DeleteAudioReq(
                            items=[
                                uburnode_somni_pb2.DeleteAudioItem(id=mid)
                                for mid in list(created_material_ids)
                            ]
                        )
                    )
                    for item in res.items:
                        print(
                            f"  DeleteAudio {item.id}: "
                            f"{'ok' if item.ok else item.message}"
                        )
                    created_material_ids.clear()
                if created_tag_ids:
                    res = await stub.DeleteAudioTag(
                        uburnode_somni_pb2.DeleteAudioTagReq(
                            items=[
                                uburnode_somni_pb2.DeleteAudioTagItem(id=tid)
                                for tid in list(created_tag_ids)
                            ]
                        )
                    )
                    for item in res.items:
                        print(
                            f"  DeleteAudioTag {item.id}: "
                            f"{'ok' if item.ok else item.message}"
                        )
                    created_tag_ids.clear()
                if material_id:
                    await _refresh_es()
                    gone = await stub.GetAudioByIds(
                        uburnode_somni_pb2.GetAudioByIdsReq(
                            ids=[material_id], language="zh"
                        )
                    )
                    if gone.list:
                        print("  WARN 删除后仍能按 id 取到")
                        results.append(("Verify deleted", False, "仍存在"))
                    else:
                        _ok("Verify deleted", "gone")
                        results.append(("Verify deleted", True, "gone"))
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  cleanup — {exc}")
                results.append(("cleanup", False, str(exc)))

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"合计 {len(results)} 步，失败 {len(failed)}")
    if failed:
        print("失败项: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(run())
    finally:
        print(f"耗时 {time.time() - started:.1f}s")
