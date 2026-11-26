"""租约编号与版本号门禁：旧请求的补偿/写入不能覆盖新请求的结果。

用一个会真正执行过滤条件的假集合（支持 $or、$lte、None 匹配缺失字段），
而不是只检查调用参数，这样 P1/P2 的竞争时序才能被复现。
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.server.somni.audio import tag_writes, writes


class _Cursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    async def to_list(self, length: int | None = None) -> list[dict]:
        return copy.deepcopy(self._docs)


class _Collection:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = [copy.deepcopy(doc) for doc in docs]

    def _match(self, doc: dict, query: dict) -> bool:
        for key, cond in query.items():
            if key == "$or":
                if not any(self._match(doc, sub) for sub in cond):
                    return False
            elif isinstance(cond, dict) and cond and all(k.startswith("$") for k in cond):
                value = doc.get(key)
                for op, arg in cond.items():
                    if op != "$lte":
                        raise AssertionError(f"假集合不支持 {op}")
                    if value is None or value > arg:
                        return False
            elif doc.get(key) != cond:
                return False
        return True

    def _find(self, query: dict) -> dict | None:
        return next((doc for doc in self.docs if self._match(doc, query)), None)

    @staticmethod
    def _apply(doc: dict, update: dict) -> None:
        doc.update(update.get("$set", {}))
        for key in update.get("$unset", {}):
            doc.pop(key, None)
        for key, delta in update.get("$inc", {}).items():
            doc[key] = doc.get(key, 0) + delta

    async def find_one(self, query: dict, *args: object, **kwargs: object) -> dict | None:
        doc = self._find(query)
        return copy.deepcopy(doc) if doc is not None else None

    def find(self, query: dict, *args: object, **kwargs: object) -> _Cursor:
        return _Cursor([doc for doc in self.docs if self._match(doc, query)])

    async def find_one_and_update(
        self, query: dict, update: dict, return_document: object = None, **_: object
    ) -> dict | None:
        doc = self._find(query)
        if doc is None:
            return None
        self._apply(doc, update)
        return copy.deepcopy(doc)

    async def update_one(self, query: dict, update: dict) -> SimpleNamespace:
        doc = self._find(query)
        if doc is None:
            return SimpleNamespace(matched_count=0, modified_count=0)
        self._apply(doc, update)
        return SimpleNamespace(matched_count=1, modified_count=1)

    async def replace_one(
        self, query: dict, replacement: dict, upsert: bool = False
    ) -> SimpleNamespace:
        doc = self._find(query)
        body = {key: value for key, value in replacement.items() if key != "_id"}
        if doc is None:
            if not upsert:
                return SimpleNamespace(matched_count=0)
            if any(existing["_id"] == query["_id"] for existing in self.docs):
                raise DuplicateKeyError("dup _id")
            self.docs.append({"_id": query["_id"], **body})
            return SimpleNamespace(matched_count=0)
        doc.clear()
        doc.update({"_id": query["_id"], **body})
        return SimpleNamespace(matched_count=1)

    async def delete_one(self, query: dict) -> SimpleNamespace:
        doc = self._find(query)
        if doc is None:
            return SimpleNamespace(deleted_count=0)
        self.docs.remove(doc)
        return SimpleNamespace(deleted_count=1)

    async def delete_many(self, query: dict) -> SimpleNamespace:
        return SimpleNamespace(deleted_count=0)


def _material_service(materials: _Collection) -> SimpleNamespace:
    return SimpleNamespace(
        _materials=lambda: materials,
        _locales=lambda: _Collection([]),
        _es_search=None,
    )


def _tag_service(tags: _Collection) -> SimpleNamespace:
    return SimpleNamespace(_tags=lambda: tags, _es_search=None)


def _content_tag(tag_id: ObjectId, **extra: object) -> dict:
    doc = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "旧名",
        "status": "启用",
        "parent_id": "p0",
        "version": 1,
    }
    doc.update(extra)
    return doc


# ---------- 原料：租约编号 ----------


@pytest.mark.asyncio
async def test_stale_holder_rollback_does_not_undo_newer_write() -> None:
    """P1：A（编号 5）写入后失去租约，B（编号 6）接管写入 vip=2，A 的补偿不能把 vip 改回 0。"""
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 0, "status": True}])
    service = _material_service(materials)
    before = await materials.find_one({"_id": mid})

    await writes._update_one_locked(service, str(mid), {"vip": 1}, {"vip": 1}, {}, fence=5)
    await writes._update_one_locked(service, str(mid), {"vip": 2}, {"vip": 2}, {}, fence=6)
    await writes._compensate_update_failure(
        service,
        str(mid),
        before_material=before,
        before_locale=None,
        wrote_locale=None,
        locale_language="",
        touched_material=True,
        touched_locale=False,
        fence=5,
    )

    doc = await materials.find_one({"_id": mid})
    assert doc["vip"] == 2
    assert doc["fence_token"] == 6


@pytest.mark.asyncio
async def test_stale_fence_write_is_rejected_and_document_untouched() -> None:
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 2, "status": True, "fence_token": 6}])
    service = _material_service(materials)

    with pytest.raises(writes.StaleLeaseError):
        await writes._update_one_locked(service, str(mid), {"vip": 1}, {"vip": 1}, {}, fence=5)

    doc = await materials.find_one({"_id": mid})
    assert doc["vip"] == 2
    assert doc["fence_token"] == 6


@pytest.mark.asyncio
async def test_current_owner_rolls_back_and_keeps_its_fence() -> None:
    """仍是最后写入者的请求正常回滚，回滚后编号保持本次租约的值，不能倒退成快照里的旧值。"""
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 0, "status": True, "fence_token": 4}])
    service = _material_service(materials)
    before = await materials.find_one({"_id": mid})

    await writes._update_one_locked(service, str(mid), {"vip": 1}, {"vip": 1}, {}, fence=5)
    await writes._compensate_update_failure(
        service,
        str(mid),
        before_material=before,
        before_locale=None,
        wrote_locale=None,
        locale_language="",
        touched_material=True,
        touched_locale=False,
        fence=5,
    )

    doc = await materials.find_one({"_id": mid})
    assert doc["vip"] == 0
    assert doc["fence_token"] == 5


@pytest.mark.asyncio
async def test_stale_delete_is_rejected_without_removing_document() -> None:
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 2, "status": True, "fence_token": 6}])
    service = _material_service(materials)

    with pytest.raises(writes.StaleLeaseError):
        await writes._delete_one_locked(service, str(mid), mid, fence=5)

    assert await materials.find_one({"_id": mid}) is not None


@pytest.mark.asyncio
async def test_es_material_patch_is_guarded_by_fence() -> None:
    client = SimpleNamespace(update=AsyncMock())

    await writes._patch_document(
        client,
        index="somni_audio_materials",
        doc_id="m1",
        fields={"vip": 1},
        remove=[],
        guard=("fence_token", 5),
    )

    script = client.update.await_args.kwargs["script"]
    assert "ctx.op = 'noop'" in script["source"]
    assert script["params"]["guard_field"] == "fence_token"
    assert script["params"]["guard_value"] == 5


# ---------- 标签：版本号 CAS 与字段级撤销 ----------


@pytest.mark.asyncio
async def test_tag_update_with_stale_version_is_rejected(monkeypatch) -> None:
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", version=2, name="新名")])
    service = _tag_service(tags)
    stale = _content_tag(tid, parent_id="p0", version=1)
    original = tags.find_one

    async def _stale_read(query: dict, *args: object, **kwargs: object) -> dict:
        tags.find_one = original
        return copy.deepcopy(stale)

    tags.find_one = _stale_read
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())

    results = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"parent_id": "p1"}}]
    )

    assert results[0]["ok"] is False
    assert "已被其他请求修改" in results[0]["message"]
    doc = await tags.find_one({"_id": tid})
    assert doc["parent_id"] == "p0"
    assert doc["version"] == 2


@pytest.mark.asyncio
async def test_failed_refresh_does_not_undo_concurrent_parent_change(monkeypatch) -> None:
    """P2：A 改名并刷新素材失败；B 在此期间把 parent_id 改成 p1 并成功。
    A 的补偿只撤销自己的改名，B 的 parent_id 必须保留。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid)])
    service = _tag_service(tags)
    calls: list[set[str]] = []

    async def _refresh(_service, codes, tag_ids=None):
        calls.append(set(codes))
        if len(calls) == 1:
            b = await tag_writes.update_audio_tags(
                service, [{"id": str(tid), "fields": {"parent_id": "p1"}}]
            )
            assert b[0]["ok"] is True
            raise RuntimeError("素材刷新失败")

    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", _refresh)

    results = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"name": "新名"}}]
    )

    assert results[0]["ok"] is False
    doc = await tags.find_one({"_id": tid})
    assert doc["parent_id"] == "p1"
    assert doc["name"] == "旧名"


@pytest.mark.asyncio
async def test_tag_compensation_restores_whole_doc_when_version_still_matches() -> None:
    tid = ObjectId()
    before = _content_tag(tid, name="旧名", parent_id="p0", version=1)
    tags = _Collection([_content_tag(tid, name="新名", parent_id="p1", version=2)])
    service = _tag_service(tags)

    await tag_writes._compensate_update_failure(
        service,
        str(tid),
        before=before,
        after=None,
        written_version=2,
        written_fields={"name": "新名", "parent_id": "p1"},
    )

    doc = await tags.find_one({"_id": tid})
    assert doc["name"] == "旧名"
    assert doc["parent_id"] == "p0"
    assert doc["version"] == 3


@pytest.mark.asyncio
async def test_es_tag_patch_is_guarded_by_version() -> None:
    client = SimpleNamespace(update=AsyncMock())

    await tag_writes._patch_document(
        client,
        index="somni_audio_tag_dictionary",
        doc_id="t1",
        fields={"name": "x"},
        remove=[],
        guard=("version", 3),
    )

    script = client.update.await_args.kwargs["script"]
    assert "ctx.op = 'noop'" in script["source"]
    assert script["params"]["guard_value"] == 3
