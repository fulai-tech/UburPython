"""租约编号与版本号门禁：旧请求的补偿/写入不能覆盖新请求的结果。

用一个会真正执行过滤条件的假集合（支持 $or、$lte、None 匹配缺失字段），
而不是只检查调用参数，这样 P1/P2 的竞争时序才能被复现。
"""

from __future__ import annotations

import asyncio
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

    def __aiter__(self):
        async def _gen():
            for doc in self._docs:
                yield copy.deepcopy(doc)

        return _gen()


class _Collection:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = [copy.deepcopy(doc) for doc in docs]

    @staticmethod
    def _get(doc: dict, path: str) -> object:
        cur: object = doc
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    def _match(self, doc: dict, query: dict) -> bool:
        for key, cond in query.items():
            if key == "$or":
                if not any(self._match(doc, sub) for sub in cond):
                    return False
            elif isinstance(cond, dict) and cond and all(k.startswith("$") for k in cond):
                value = self._get(doc, key)
                for op, arg in cond.items():
                    if op == "$lte":
                        if value is None or value > arg:
                            return False
                    elif op == "$in":
                        if value not in arg:
                            return False
                    else:
                        raise AssertionError(f"假集合不支持 {op}")
            elif self._get(doc, key) != cond:
                return False
        return True

    def _find(self, query: dict) -> dict | None:
        return next((doc for doc in self.docs if self._match(doc, query)), None)

    async def insert_one(self, doc: dict) -> SimpleNamespace:
        if any(existing["_id"] == doc["_id"] for existing in self.docs):
            raise DuplicateKeyError("dup _id")
        self.docs.append(copy.deepcopy(doc))
        return SimpleNamespace(inserted_id=doc["_id"])

    @staticmethod
    def _apply(doc: dict, update: dict) -> None:
        for path, value in update.get("$set", {}).items():
            parts = path.split(".")
            cur = doc
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
        for path in update.get("$unset", {}):
            parts = path.split(".")
            cur = doc
            for part in parts[:-1]:
                cur = cur.get(part)
                if not isinstance(cur, dict):
                    break
            else:
                cur.pop(parts[-1], None)
        for key, delta in update.get("$inc", {}).items():
            doc[key] = doc.get(key, 0) + delta

    async def find_one(self, query: dict, *args: object, **kwargs: object) -> dict | None:
        doc = self._find(query)
        return copy.deepcopy(doc) if doc is not None else None

    def find(self, query: dict, *args: object, **kwargs: object) -> _Cursor:
        return _Cursor([doc for doc in self.docs if self._match(doc, query)])

    async def find_one_and_update(
        self,
        query: dict,
        update: dict,
        return_document: object = None,
        upsert: bool = False,
        **_: object,
    ) -> dict | None:
        doc = self._find(query)
        if doc is None:
            if not upsert:
                return None
            doc = {"_id": query.get("_id") or ObjectId()}
            for key, value in query.items():
                if not key.startswith("$") and not isinstance(value, dict):
                    doc[key] = value
            doc.update(update.get("$setOnInsert", {}))
            self._apply(doc, update)
            self.docs.append(doc)
            return copy.deepcopy(doc)
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
        matched = [doc for doc in self.docs if self._match(doc, query)]
        for doc in matched:
            self.docs.remove(doc)
        return SimpleNamespace(deleted_count=len(matched))


def _material_service(
    materials: _Collection, locales: _Collection | None = None
) -> SimpleNamespace:
    locale_coll = locales if locales is not None else _Collection([])
    return SimpleNamespace(
        _materials=lambda: materials,
        _locales=lambda: locale_coll,
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
async def test_stale_holder_rollback_does_not_undo_newer_locale_title() -> None:
    """P1-1：租约 6 成功更新 vip=2 与中文标题后，租约 5 补偿不能把语言行替换回旧快照。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 0, "status": True}])
    locales = _Collection(
        [{"_id": lid, "material_id": str(mid), "language": "zh", "audio_name": "旧标题"}]
    )
    service = _material_service(materials, locales)
    before_material = await materials.find_one({"_id": mid})
    before_locale = await locales.find_one({"_id": lid})

    await writes._update_one_locked(
        service,
        str(mid),
        {"vip": 1, "language": "zh", "audio_name": "租约5标题"},
        {"vip": 1},
        {"language": "zh", "audio_name": "租约5标题"},
        fence=5,
    )
    wrote_locale_5 = await locales.find_one({"_id": lid})
    await writes._update_one_locked(
        service,
        str(mid),
        {"vip": 2, "language": "zh", "audio_name": "租约6新标题"},
        {"vip": 2},
        {"language": "zh", "audio_name": "租约6新标题"},
        fence=6,
    )
    await writes._compensate_update_failure(
        service,
        str(mid),
        before_material=before_material,
        before_locale=before_locale,
        wrote_locale=wrote_locale_5,
        locale_language="zh",
        touched_material=True,
        touched_locale=True,
        fence=5,
    )

    material_doc = await materials.find_one({"_id": mid})
    locale_doc = await locales.find_one({"_id": lid})
    assert material_doc["vip"] == 2
    assert material_doc["fence_token"] == 6
    assert locale_doc["audio_name"] == "租约6新标题"
    assert locale_doc["fence_token"] == 6


@pytest.mark.asyncio
async def test_stale_locale_only_update_and_new_locale_rollback_respect_fence() -> None:
    """P1-1：旧租约仅改语言行会被拒绝；初始无语言行时旧租约补偿也不能删掉新租约写的语言行。"""
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 0, "status": True}])
    locales = _Collection([])
    service = _material_service(materials, locales)
    before_material = await materials.find_one({"_id": mid})

    await writes._update_one_locked(
        service,
        str(mid),
        {"language": "en", "audio_name": "Draft 5"},
        {},
        {"language": "en", "audio_name": "Draft 5"},
        fence=5,
    )
    wrote_locale_5 = await locales.find_one({"material_id": str(mid), "language": "en"})
    await writes._update_one_locked(
        service,
        str(mid),
        {"language": "en", "audio_name": "Final 6"},
        {},
        {"language": "en", "audio_name": "Final 6"},
        fence=6,
    )

    with pytest.raises(writes.StaleLeaseError):
        await writes._update_one_locked(
            service,
            str(mid),
            {"language": "en", "audio_name": "Late 5"},
            {},
            {"language": "en", "audio_name": "Late 5"},
            fence=5,
        )

    await writes._compensate_update_failure(
        service,
        str(mid),
        before_material=before_material,
        before_locale=None,
        wrote_locale=wrote_locale_5,
        locale_language="en",
        touched_material=False,
        touched_locale=True,
        fence=5,
    )

    locale_doc = await locales.find_one({"material_id": str(mid), "language": "en"})
    assert locale_doc is not None
    assert locale_doc["audio_name"] == "Final 6"
    assert locale_doc["fence_token"] == 6


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
async def test_stale_delete_compensation_does_not_resurrect_deleted_material() -> None:
    """P1-2：租约 6 删除成功后，租约 5 收到连接异常开始补偿，不能把已删除的原料与语言行复活。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 1, "status": True}])
    locales = _Collection(
        [{"_id": lid, "material_id": str(mid), "language": "zh", "audio_name": "雨声"}]
    )
    es_client = SimpleNamespace(
        index=AsyncMock(),
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    service = _material_service(materials, locales)
    service._es_search = SimpleNamespace(
        audio_index="somni_audio_materials",
        material_locales_index="somni_audio_material_locales",
        _client=es_client,
    )
    before_material = await materials.find_one({"_id": mid})
    before_rows = await locales.find({"material_id": str(mid)}).to_list(length=None)

    await writes._delete_one_locked(service, str(mid), mid, fence=6)
    assert await materials.find_one({"_id": mid}) is None
    assert await locales.find_one({"_id": lid}) is None
    es_client.index.reset_mock()

    await writes._compensate_delete_failure(
        service,
        before_material,
        before_rows,
        locales_deleted=False,
        fence=5,
    )
    await writes._compensate_delete_failure(
        service,
        before_material,
        before_rows,
        locales_deleted=True,
        fence=5,
    )

    assert await materials.find_one({"_id": mid}) is None
    assert await locales.find_one({"_id": lid}) is None
    es_client.index.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_delete_failure_still_restores_when_not_superseded() -> None:
    """P1-2 对照：本租约（6）删除中途 ES 失败时，自己的补偿仍可恢复原料，且编号保持 6。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 1, "status": True, "fence_token": 4}])
    locales = _Collection(
        [{"_id": lid, "material_id": str(mid), "language": "zh", "audio_name": "雨声"}]
    )
    es_client = SimpleNamespace(
        index=AsyncMock(),
        update=AsyncMock(),
        delete=AsyncMock(side_effect=RuntimeError("es delete down")),
    )
    service = _material_service(materials, locales)
    service._es_search = SimpleNamespace(
        audio_index="somni_audio_materials",
        material_locales_index="somni_audio_material_locales",
        _client=es_client,
    )

    with pytest.raises(RuntimeError, match="es delete down"):
        await writes._delete_one_locked(service, str(mid), mid, fence=6)

    restored_mat = await materials.find_one({"_id": mid})
    restored_loc = await locales.find_one({"_id": lid})
    assert restored_mat is not None
    assert restored_mat["fence_token"] == 6
    assert restored_loc is not None
    assert restored_loc["fence_token"] == 6


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
    """P2：A 改名并刷新素材失败；B 在此期间把 parent_id 改成 p1。
    同一标签的写入串行：B 排在 A 的补偿之后执行，A 只撤销自己的改名，B 的 parent_id 保留。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid)])
    service = _tag_service(tags)
    calls: list[set[str]] = []
    b_tasks: list[asyncio.Task] = []

    async def _refresh(_service, codes, tag_ids=None):
        calls.append(set(codes))
        if len(calls) == 1:
            b_tasks.append(
                asyncio.create_task(
                    tag_writes.update_audio_tags(
                        service, [{"id": str(tid), "fields": {"parent_id": "p1"}}]
                    )
                )
            )
            await asyncio.sleep(0.01)
            raise RuntimeError("素材刷新失败")

    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", _refresh)

    results = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"name": "新名"}}]
    )
    b_results = await b_tasks[0]

    assert results[0]["ok"] is False
    assert b_results[0]["ok"] is True
    doc = await tags.find_one({"_id": tid})
    assert doc["parent_id"] == "p1"
    assert doc["name"] == "旧名"


@pytest.mark.asyncio
async def test_tag_compensation_reverts_only_fields_still_owned_by_this_write() -> None:
    """补偿只撤销仍由本笔（op-A）占有的字段；parent_id 已被 op-B 改写，必须保留。"""
    tid = ObjectId()
    before = _content_tag(tid, name="旧名", parent_id="p0", version=1)
    tags = _Collection(
        [
            _content_tag(
                tid,
                name="新名",
                parent_id="p1",
                version=2,
                _ops={"name": "op-A", "parent_id": "op-B"},
            )
        ]
    )
    service = _tag_service(tags)

    await tag_writes._compensate_update_failure(
        service,
        str(tid),
        before=before,
        after=None,
        op="op-A",
        written_fields={"name": "新名", "parent_id": "p1"},
    )

    doc = await tags.find_one({"_id": tid})
    assert doc["name"] == "旧名"
    assert doc["parent_id"] == "p1"
    assert "name" not in doc.get("_ops", {})


@pytest.mark.asyncio
async def test_uncommitted_tag_update_error_does_not_rollback_concurrent_version_2(
    monkeypatch,
) -> None:
    """P1-3：A、B 都读到版本 1，B 成功把 parent_id 改为 p1（版本 2），
    A 的写入未落库并报连接异常，不能把版本 2 当作自己的写入回滚回 p0。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", name="旧名", version=1)])
    es_client = SimpleNamespace(index=AsyncMock(), update=AsyncMock(), delete=AsyncMock())
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(
        tag_dictionary_index="somni_audio_tag_dictionary",
        refresh_audio_index=AsyncMock(),
        clear_content_tag_vectors_cache=lambda: None,
        _client=es_client,
    )
    service._encoder = SimpleNamespace(encode_one=AsyncMock(return_value=[0.1, 0.2]))
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())

    stale_v1 = await tags.find_one({"_id": tid})
    b_res = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"parent_id": "p1"}}]
    )
    assert b_res[0]["ok"] is True

    orig_find_one = tags.find_one
    orig_fau = tags.find_one_and_update

    async def _a_reads_v1(query: dict, *args: object, **kwargs: object):
        tags.find_one = orig_find_one
        return copy.deepcopy(stale_v1)

    async def _a_fau_network_error(*_args: object, **_kwargs: object):
        tags.find_one_and_update = orig_fau
        raise ConnectionError("mongo connection lost before commit")

    tags.find_one = _a_reads_v1
    tags.find_one_and_update = _a_fau_network_error

    a_res = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"parent_id": "p1", "name": "A未落库名"}}]
    )

    assert a_res[0]["ok"] is False
    doc = await tags.find_one({"_id": tid})
    assert doc["parent_id"] == "p1"
    assert doc["name"] == "旧名"


@pytest.mark.asyncio
async def test_tag_delete_error_does_not_overwrite_concurrent_successful_edit(
    monkeypatch,
) -> None:
    """P1-3：A 删除前读到版本 1 旧快照，B 成功编辑成版本 2 新名称和 p1，
    A 删除报连接异常进入补偿时，不能用旧快照覆写 B 的成功编辑。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", name="旧名", version=1)])
    es_client = SimpleNamespace(index=AsyncMock(), update=AsyncMock(), delete=AsyncMock())
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(
        tag_dictionary_index="somni_audio_tag_dictionary",
        refresh_audio_index=AsyncMock(),
        clear_content_tag_vectors_cache=lambda: None,
        _client=es_client,
    )
    service._encoder = SimpleNamespace(encode_one=AsyncMock(return_value=[0.1, 0.2]))
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())

    stale_v1 = await tags.find_one({"_id": tid})
    b_res = await tag_writes.update_audio_tags(
        service, [{"id": str(tid), "fields": {"name": "B新名", "parent_id": "p1"}}]
    )
    assert b_res[0]["ok"] is True
    es_client.index.reset_mock()

    orig_find_one = tags.find_one
    orig_delete_one = tags.delete_one

    async def _a_reads_v1(query: dict, *args: object, **kwargs: object):
        tags.find_one = orig_find_one
        return copy.deepcopy(stale_v1)

    async def _a_delete_network_error(*_args: object, **_kwargs: object):
        tags.delete_one = orig_delete_one
        raise ConnectionError("mongo delete connection lost")

    tags.find_one = _a_reads_v1
    tags.delete_one = _a_delete_network_error

    del_res = await tag_writes.delete_audio_tags(service, [{"id": str(tid)}])

    assert del_res[0]["ok"] is False
    doc = await tags.find_one({"_id": tid})
    assert doc["name"] == "B新名"
    assert doc["parent_id"] == "p1"
    es_client.index.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_tag_delete_compensation_does_not_resurrect_deleted_tag(
    monkeypatch,
) -> None:
    """P1-3：A（租约 1）删除前先记墓碑；随后 B（租约 2）删除成功。
    A 的删除补偿不能把已被 B 删除的标签插回去。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", name="旧名", version=1)])
    tombstones = _Collection([])
    service = _tag_service(tags)
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())
    before = await tags.find_one({"_id": tid})

    key = f"tag:{tid}"
    await writes._record_delete_tombstone(service, key, 1)  # A 的墓碑
    await writes._record_delete_tombstone(service, key, 2)  # B 的墓碑（更新）
    await tags.delete_one({"_id": tid})  # B 删除成功

    await tag_writes._compensate_delete_failure(service, str(tid), before, fence=1)

    assert await tags.find_one({"_id": tid}) is None


@pytest.mark.asyncio
async def test_tag_delete_compensation_restores_when_no_newer_delete(monkeypatch) -> None:
    """对照：没有更新的删除接管时，本租约删除失败的补偿仍能把标签插回。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", name="旧名", version=1)])
    tombstones = _Collection([])
    service = _tag_service(tags)
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}
    service._es_search = None
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())
    before = await tags.find_one({"_id": tid})

    await writes._record_delete_tombstone(service, f"tag:{tid}", 1)
    await tags.delete_one({"_id": tid})

    await tag_writes._compensate_delete_failure(service, str(tid), before, fence=1)

    restored = await tags.find_one({"_id": tid})
    assert restored is not None
    assert restored["name"] == "旧名"


@pytest.mark.asyncio
async def test_concurrent_rename_and_parent_change_keep_es_in_sync(monkeypatch) -> None:
    """P1-4：A 改名并等待向量编码期间，B 改父标签。B 必须排在 A 的 ES 补丁之后，
    否则 B 的版本 3 补丁会让 A 的版本 2 补丁被跳过，ES 停留在旧名与旧向量。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, parent_id="p0", name="旧名", version=1)])
    es = _FakeTagEs()
    es.docs[str(tid)] = {"name": "旧名", "name_vector": [0.1, 0.1], "parent_id": "p0", "version": 1}
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowEncoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            if text == "A新名":
                started.set()
                await release.wait()
                return [0.9, 0.8]
            return [0.1, 0.1]

    service = _tag_service(tags)
    service._es_search = SimpleNamespace(
        tag_dictionary_index="somni_audio_tag_dictionary",
        refresh_audio_index=AsyncMock(),
        clear_content_tag_vectors_cache=lambda: None,
        _client=es,
    )
    service._encoder = _SlowEncoder()
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())

    a_task = asyncio.create_task(
        tag_writes.update_audio_tags(service, [{"id": str(tid), "fields": {"name": "A新名"}}])
    )
    await started.wait()
    b_task = asyncio.create_task(
        tag_writes.update_audio_tags(service, [{"id": str(tid), "fields": {"parent_id": "p1"}}])
    )
    await asyncio.sleep(0.05)
    assert es.calls == [], "B 不应在 A 的 ES 补丁完成前写入 ES"

    release.set()
    a_res, b_res = await asyncio.gather(a_task, b_task)

    assert a_res[0]["ok"] is True
    assert b_res[0]["ok"] is True
    es_doc = es.docs[str(tid)]
    assert es_doc["name"] == "A新名"
    assert es_doc["name_vector"] == [0.9, 0.8]
    assert es_doc["parent_id"] == "p1"


class _FakeTagEs:
    """按 ES 语义模拟标签索引：局部更新支持版本守卫（守卫字段更大则 noop）。"""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.calls: list[dict] = []

    async def update(self, *, index: str, id: str, script=None, doc=None, **_: object) -> None:
        self.calls.append({"index": index, "id": id})
        current = self.docs.setdefault(id, {})
        if script is None:
            current.update(doc or {})
            return
        params = script["params"]
        guard = params.get("guard_field")
        if guard is not None:
            seen = current.get(guard)
            if seen is not None and seen > params["guard_value"]:
                return
        current.update(params["fields"])
        for name in params["remove"]:
            current.pop(name, None)
        if guard is not None:
            current[guard] = params["guard_value"]

    async def index(self, *, index: str, id: str, document: dict, **_: object) -> None:
        self.calls.append({"index": index, "id": id})
        self.docs[id] = dict(document)

    async def delete(self, *, index: str, id: str, **_: object) -> None:
        self.calls.append({"index": index, "id": id})
        self.docs.pop(id, None)

    async def delete_by_query(self, *, index: str, query: dict, **_: object) -> dict:
        self.calls.append({"index": index, "query": query})
        material_id = (query.get("term") or {}).get("material_id")
        removed = [
            doc_id
            for doc_id, doc in self.docs.items()
            if doc.get("material_id") == material_id
        ]
        for doc_id in removed:
            self.docs.pop(doc_id, None)
        return {"deleted": len(removed)}


# ---------- 审查补充：删除墓碑与补偿边界 ----------


@pytest.mark.asyncio
async def test_retry_delete_of_already_removed_material_blocks_stale_compensation() -> None:
    """租约 5 已删除原料但进入补偿；租约 6 重试删除时库里已无此原料，
    也必须记下墓碑，否则租约 5 的补偿会把原料复活。"""
    mid = ObjectId()
    materials = _Collection([])
    service = _material_service(materials, _Collection([]))
    tombstones = _Collection([{"_id": str(mid), "fence_token": 5, "deleted_at": None}])
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}

    with pytest.raises(ValueError):
        await writes._delete_one_locked(service, str(mid), mid, fence=6)
    await writes._compensate_delete_failure(
        service,
        {"_id": mid, "vip": 1, "status": True},
        [],
        locales_deleted=False,
        fence=5,
    )

    assert await materials.find_one({"_id": mid}) is None


@pytest.mark.asyncio
async def test_retry_delete_purges_leftover_locale_and_es_when_material_is_gone() -> None:
    """原料已删、语言行和 ES 还在时，更新的删除要清掉残留，并挡住旧补偿复活。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([])
    locales = _Collection(
        [
            {
                "_id": lid,
                "material_id": str(mid),
                "language": "zh",
                "audio_name": "雨声",
                "fence_token": 5,
            }
        ]
    )
    es = _FakeTagEs()
    es.docs[str(mid)] = {"vip": 1}
    es.docs[str(lid)] = {"audio_name": "雨声"}
    service = _material_service(materials, locales)
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )
    tombstones = _Collection(
        [{"_id": str(mid), "fence_token": 5, "deleted_at": None}]
    )
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}

    await writes._delete_one_locked(service, str(mid), mid, fence=6)

    assert await materials.find_one({"_id": mid}) is None
    assert await locales.find_one({"_id": lid}) is None
    assert str(mid) not in es.docs
    assert str(lid) not in es.docs
    await writes._compensate_delete_failure(
        service,
        {"_id": mid, "vip": 1, "status": True},
        [
            {
                "_id": lid,
                "material_id": str(mid),
                "language": "zh",
                "audio_name": "雨声",
            }
        ],
        locales_deleted=True,
        fence=5,
    )
    assert await materials.find_one({"_id": mid}) is None
    assert await locales.find_one({"_id": lid}) is None


@pytest.mark.asyncio
async def test_retry_delete_purges_locale_es_when_locale_row_is_gone() -> None:
    """语言行已经删掉、语言索引还在时，重试仍要按原料 id 清掉检索文档。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([])
    locales = _Collection([])
    es = _FakeTagEs()
    es.docs[str(mid)] = {"vip": 1}
    es.docs[str(lid)] = {"audio_name": "雨声", "material_id": str(mid)}
    service = _material_service(materials, locales)
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )
    tombstones = _Collection([{"_id": str(mid), "fence_token": 5, "deleted_at": None}])
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}

    await writes._delete_one_locked(service, str(mid), mid, fence=6)

    assert str(mid) not in es.docs
    assert str(lid) not in es.docs
    assert await materials.find_one({"_id": mid}) is None


@pytest.mark.asyncio
async def test_retry_tag_delete_removes_leftover_es_when_tag_is_gone(monkeypatch) -> None:
    """标签文档已经删掉、索引还在时，重试要删掉索引，并挡住旧补偿插回。"""
    tid = ObjectId()
    tags = _Collection([])
    es = _FakeTagEs()
    es.docs[str(tid)] = {"name": "雨声", "version": 1}
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)
    tombstones = _Collection(
        [{"_id": f"tag:{tid}", "fence_token": 1, "deleted_at": None}]
    )
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())

    await tag_writes._delete_one_locked(service, str(tid), tid, fence=2)

    assert str(tid) not in es.docs
    await tag_writes._compensate_delete_failure(
        service,
        str(tid),
        _content_tag(tid, name="雨声"),
        fence=1,
    )
    assert await tags.find_one({"_id": tid}) is None


@pytest.mark.asyncio
async def test_stale_locale_rollback_still_reindexes_when_material_taken_over(
    monkeypatch,
) -> None:
    """租约 5 只改了语言行并已回滚成功；租约 6 已接管原料。
    回滚后的语言行仍是本租约写入的那一版，必须重刷语言 ES 索引，不能因原料已被接管而跳过。"""
    mid = ObjectId()
    lid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 2, "status": True, "fence_token": 6}])
    locales = _Collection(
        [
            {
                "_id": lid,
                "material_id": str(mid),
                "language": "en",
                "audio_name": "Broken A",
                "fence_token": 5,
            }
        ]
    )
    service = _material_service(materials, locales)
    reindex = AsyncMock()
    monkeypatch.setattr(writes, "_reindex_material_locales", reindex)

    await writes._compensate_update_failure(
        service,
        str(mid),
        before_material={"_id": mid, "vip": 0, "status": True},
        before_locale={"_id": lid, "material_id": str(mid), "language": "en", "audio_name": "Old"},
        wrote_locale={"_id": lid},
        locale_language="en",
        touched_material=False,
        touched_locale=True,
        fence=5,
    )

    assert (await locales.find_one({"_id": lid}))["audio_name"] == "Old"
    reindex.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_tag_delete_of_already_removed_tag_blocks_stale_compensation() -> None:
    """标签已被租约 5 删除；租约 6 重试时发现不存在，也要记墓碑，阻止租约 5 的补偿插回。"""
    tid = ObjectId()
    tags = _Collection([])
    service = _tag_service(tags)
    tombstones = _Collection([{"_id": f"tag:{tid}", "fence_token": 5, "deleted_at": None}])
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}

    with pytest.raises(ValueError):
        await tag_writes._delete_one_locked(service, str(tid), tid, fence=6)
    await tag_writes._compensate_delete_failure(
        service,
        str(tid),
        _content_tag(tid, parent_id="p0", name="旧名", version=1),
        fence=5,
    )

    assert await tags.find_one({"_id": tid}) is None


@pytest.mark.asyncio
async def test_tag_restore_is_withdrawn_when_newer_delete_lands_after_insert(
    monkeypatch,
) -> None:
    """租约 1 的补偿在检查后插回标签；随后租约 2 删除成功并记墓碑。
    插入后必须复检，并撤回自己刚插回的文档。"""
    tid = ObjectId()
    tags = _Collection([])
    tombstones = _Collection([])
    service = _tag_service(tags)
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}
    monkeypatch.setattr(tag_writes, "refresh_locales_for_content_codes", AsyncMock())
    original_insert = tags.insert_one

    async def _insert_then_newer_delete(doc):
        result = await original_insert(doc)
        await writes._record_delete_tombstone(service, f"tag:{tid}", 2)
        return result

    tags.insert_one = _insert_then_newer_delete

    await tag_writes._compensate_delete_failure(
        service,
        str(tid),
        _content_tag(tid, parent_id="p0", name="旧名", version=1),
        fence=1,
    )

    assert await tags.find_one({"_id": tid}) is None


@pytest.mark.asyncio
async def test_tombstone_store_fails_closed_when_db_unavailable() -> None:
    """启用租约（有 fence）时若墓碑库不可用，必须抛出，不能静默退回进程内存。"""

    def _broken_db():
        raise RuntimeError("db down")

    service = SimpleNamespace(_db=_broken_db, _materials=lambda: _Collection([]))

    with pytest.raises(RuntimeError):
        writes._delete_tombstones(service)


def test_fallback_tombstone_store_is_stable_per_service() -> None:
    """无 _db 的服务（仅测试）：同一服务多次取到的墓碑必须是同一份。"""
    service = SimpleNamespace(_materials=lambda: _Collection([]))

    assert writes._delete_tombstones(service) is writes._delete_tombstones(service)


# ---------- 补偿整份写 ES：不能盖掉更新的成功结果 ----------


@pytest.mark.asyncio
async def test_tag_reindex_after_encode_does_not_overwrite_newer_version() -> None:
    """算向量期间标签已被写成版本 3。旧快照（版本 2）不能把 ES 盖回去。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="B新名", version=3, parent_id="p1")])
    es = _FakeTagEs()
    es.docs[str(tid)] = {"name": "B新名", "version": 3, "parent_id": "p1"}
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [0.9, 0.8]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="A旧名", version=2))

    assert es.docs[str(tid)]["name"] == "B新名"
    assert es.docs[str(tid)]["version"] == 3


@pytest.mark.asyncio
async def test_version_zero_reindex_does_not_overwrite_newer_tag() -> None:
    """新建标签版本是 0。算向量期间被更新到版本 1 后，不能把创建时的文档盖回 ES。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="已更新", version=1, parent_id="p1")])
    es = _FakeTagEs()
    es.docs[str(tid)] = {"name": "已更新", "version": 1, "parent_id": "p1"}
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [0.2, 0.2]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="刚创建", version=0))

    assert es.docs[str(tid)]["name"] == "已更新"
    assert es.docs[str(tid)]["version"] == 1


@pytest.mark.asyncio
async def test_version_zero_index_repairs_es_when_update_lands_during_index() -> None:
    """版本 0 的整份写入落地时，标签已被更新。ES 必须改成库里的新字段。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="刚创建", version=0)])
    es = _FakeTagEs()
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)
    original = es.index

    async def _index_then_update(*, index: str, id: str, document: dict, **_: object) -> None:
        await original(index=index, id=id, document=document)
        tags.docs[0]["name"] = "已更新"
        tags.docs[0]["version"] = 1

    es.index = _index_then_update

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [0.2, 0.2]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="刚创建", version=0))

    assert es.docs[str(tid)]["name"] == "已更新"
    assert es.docs[str(tid)]["version"] == 1


@pytest.mark.asyncio
async def test_create_compensation_does_not_delete_tag_updated_by_newer_lease() -> None:
    """创建租约被接管并更新成功后，创建补偿不能按 _id 删掉新文档，也不能删掉 ES。"""
    tid = ObjectId()
    tags = _Collection(
        [
            _content_tag(
                tid,
                name="已更新",
                version=1,
                fence_token=2,
                locale_refresh_pending=True,
            )
        ]
    )
    es = _FakeTagEs()
    es.docs[str(tid)] = {"name": "已更新", "version": 1}
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)

    await tag_writes._compensate_create_failure(
        service,
        str(tid),
        _content_tag(tid, name="刚创建", version=0, fence_token=1),
        fence=1,
    )

    doc = await tags.find_one({"_id": tid})
    assert doc["name"] == "已更新"
    assert doc["locale_refresh_pending"] is True
    assert es.docs[str(tid)]["name"] == "已更新"


@pytest.mark.asyncio
async def test_create_keeps_pending_flag_when_update_wins_during_encode() -> None:
    """算向量期间标签已被更新。创建不能清掉新请求的待刷新标记。"""
    tags = _Collection([])
    es = _FakeTagEs()
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            tags.docs[0]["name"] = "已更新"
            tags.docs[0]["version"] = 1
            tags.docs[0]["locale_refresh_pending"] = True
            return [0.2, 0.2]

    service._encoder = _Encoder()
    tag_id = await tag_writes._create_one(
        service,
        {
            "type": "content_form",
            "code": "steady_rain",
            "language": "zh",
            "name": "刚创建",
        },
    )

    doc = await tags.find_one({"_id": ObjectId(tag_id)})
    assert doc["name"] == "已更新"
    assert doc["version"] == 1
    assert doc["locale_refresh_pending"] is True


@pytest.mark.asyncio
async def test_tag_reindex_does_not_resurrect_es_after_delete() -> None:
    """算向量期间标签已从库里删除。整份写入不能把 ES 文档重新建出来。"""
    tid = ObjectId()
    es = _FakeTagEs()
    service = _tag_service(_Collection([]))
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [0.1, 0.1]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="旧名", version=1))

    assert str(tid) not in es.docs


@pytest.mark.asyncio
async def test_material_reindex_skips_when_fence_no_longer_matches() -> None:
    """原料编号已是 6 时，编号 5 的整份快照不能覆盖 ES。"""
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 2, "status": True, "fence_token": 6}])
    es = _FakeTagEs()
    es.docs[str(mid)] = {"vip": 2, "fence_token": 6, "status": True}
    service = _material_service(materials)
    service._es_search = SimpleNamespace(audio_index="materials", _client=es)

    await writes._index_material_document(
        service,
        str(mid),
        {"vip": 1, "status": True, "fence_token": 5},
    )

    assert es.docs[str(mid)]["vip"] == 2
    assert es.docs[str(mid)]["fence_token"] == 6


@pytest.mark.asyncio
async def test_material_reindex_does_not_resurrect_deleted_material() -> None:
    mid = ObjectId()
    es = _FakeTagEs()
    service = _material_service(_Collection([]))
    service._es_search = SimpleNamespace(audio_index="materials", _client=es)

    await writes._index_material_document(
        service,
        str(mid),
        {"vip": 1, "status": True, "fence_token": 5},
    )

    assert str(mid) not in es.docs


@pytest.mark.asyncio
async def test_locale_reindex_after_encode_skips_when_row_taken_over() -> None:
    """语言行在算向量期间被编号 6 写过。编号 5 的标题不能再进入 ES。"""
    lid = ObjectId()
    locales = _Collection(
        [
            {
                "_id": lid,
                "material_id": "m",
                "language": "zh",
                "audio_name": "新标题",
                "description_text": "新",
                "fence_token": 6,
            }
        ]
    )
    es = _FakeTagEs()
    es.docs[str(lid)] = {"audio_name": "新标题", "fence_token": 6}
    service = _material_service(_Collection([]), locales)
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [1.0, 0.0]

    service._encoder = _Encoder()
    await writes._index_locale_document(
        service,
        str(lid),
        {
            "language": "zh",
            "audio_name": "旧标题",
            "description_text": "旧",
            "fence_token": 5,
        },
    )

    assert es.docs[str(lid)]["audio_name"] == "新标题"
    assert es.docs[str(lid)]["fence_token"] == 6


@pytest.mark.asyncio
async def test_material_reindex_repairs_es_when_fence_moves_during_index() -> None:
    """整份写入落地的同时编号被推进，ES 必须改成库里的新值，不能留下旧快照。"""
    mid = ObjectId()
    materials = _Collection([{"_id": mid, "vip": 1, "status": True, "fence_token": 5}])
    es = _FakeTagEs()
    service = _material_service(materials)
    service._es_search = SimpleNamespace(audio_index="materials", _client=es)
    original = es.index

    async def _index_then_takeover(*, index: str, id: str, document: dict, **_: object) -> None:
        await original(index=index, id=id, document=document)
        if materials.docs and materials.docs[0].get("fence_token") == 5:
            materials.docs[0]["fence_token"] = 6
            materials.docs[0]["vip"] = 2

    es.index = _index_then_takeover

    await writes._index_material_document(
        service,
        str(mid),
        {"vip": 1, "status": True, "fence_token": 5},
    )

    assert es.docs[str(mid)]["vip"] == 2
    assert es.docs[str(mid)]["fence_token"] == 6


@pytest.mark.asyncio
async def test_create_compensation_does_not_delete_material_taken_over_by_newer_lease() -> None:
    """创建写索引时被更高编号接管。失败补偿不能删掉新原料、新语言行和已写好的 ES。"""
    materials = _Collection([])
    locales = _Collection([])
    locks = _Collection([])
    es = _FakeTagEs()
    service = _material_service(materials, locales)
    service._db = lambda: {"somni_audio_index_locks": locks}
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )

    async def _take_over_then_fail(*, index: str, id: str, document: dict, **_: object) -> None:
        if index != "materials":
            return
        materials.docs[0]["vip"] = 2
        materials.docs[0]["fence_token"] = 6
        locales.docs[0]["audio_name"] = "新标题"
        locales.docs[0]["fence_token"] = 6
        es.docs[id] = {"vip": 2, "fence_token": 6}
        es.docs[str(locales.docs[0]["_id"])] = {"audio_name": "新标题", "fence_token": 6}
        raise RuntimeError("检索索引写入失败")

    es.index = _take_over_then_fail

    with pytest.raises(RuntimeError, match="检索索引写入失败"):
        await writes._create_one(
            service, {"audio_name": "刚创建", "language": "zh", "vip": 0}
        )

    assert materials.docs[0]["vip"] == 2
    assert materials.docs[0]["fence_token"] == 6
    assert locales.docs[0]["audio_name"] == "新标题"
    assert locales.docs[0]["fence_token"] == 6
    assert es.docs[str(materials.docs[0]["_id"])]["vip"] == 2
    assert es.docs[str(locales.docs[0]["_id"])]["audio_name"] == "新标题"


@pytest.mark.asyncio
async def test_create_compensation_still_removes_material_owned_by_this_lease() -> None:
    """创建失败且编号仍是本次租约时，原料、语言行和 ES 都要清掉。"""
    materials = _Collection([])
    locales = _Collection([])
    locks = _Collection([])
    es = _FakeTagEs()
    service = _material_service(materials, locales)
    service._db = lambda: {"somni_audio_index_locks": locks}
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )
    original = es.index

    async def _index_material_then_fail(
        *, index: str, id: str, document: dict, **_: object
    ) -> None:
        if index == "materials":
            await original(index=index, id=id, document=document)
            raise RuntimeError("检索索引写入失败")
        await original(index=index, id=id, document=document)

    es.index = _index_material_then_fail

    with pytest.raises(RuntimeError, match="检索索引写入失败"):
        await writes._create_one(
            service, {"audio_name": "刚创建", "language": "zh", "vip": 0}
        )

    assert materials.docs == []
    assert locales.docs == []
    assert es.docs == {}


@pytest.mark.asyncio
async def test_locale_repair_keeps_status_tags_and_vector_when_fence_moves_during_index() -> None:
    """编号在整份写入落地时被推进后，修复必须带上停用状态、标签名和按新文本算出的向量。"""
    mid = ObjectId()
    lid = ObjectId()
    tag_id = ObjectId()
    locales = _Collection(
        [
            {
                "_id": lid,
                "material_id": str(mid),
                "language": "zh",
                "audio_name": "旧标题",
                "fence_token": 5,
            }
        ]
    )
    materials = _Collection(
        [
            {
                "_id": mid,
                "status": False,
                "fence_token": 5,
                "content_form_tags": [{"code": "rain", "tag_id": str(tag_id)}],
            }
        ]
    )
    tags = _Collection(
        [
            {
                "_id": tag_id,
                "type": "content_form",
                "code": "rain",
                "name": "雨声",
                "language": "zh",
                "status": "启用",
            }
        ]
    )
    es = _FakeTagEs()
    service = _material_service(materials, locales)
    service._tags = lambda: tags
    service._es_search = SimpleNamespace(
        audio_index="materials",
        material_locales_index="locales",
        _client=es,
    )
    original = es.index

    async def _index_then_takeover(*, index: str, id: str, document: dict, **_: object) -> None:
        await original(index=index, id=id, document=document)
        if locales.docs and locales.docs[0].get("fence_token") == 5:
            locales.docs[0]["fence_token"] = 6
            locales.docs[0]["audio_name"] = "新标题"
            materials.docs[0]["fence_token"] = 6

    es.index = _index_then_takeover

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [float(len(text))]

    service._encoder = _Encoder()
    await writes._index_locale_document(
        service,
        str(lid),
        {
            "material_id": str(mid),
            "language": "zh",
            "audio_name": "旧标题",
            "description_text": "旧标题",
            "material_status": False,
            "fence_token": 5,
        },
    )

    doc = es.docs[str(lid)]
    assert doc["audio_name"] == "新标题"
    assert doc["description_text"] == "新标题 雨声"
    assert doc["material_status"] is False
    assert doc["fence_token"] == 6
    assert doc["description_vector"] == [float(len("新标题 雨声"))]


@pytest.mark.asyncio
async def test_tag_repair_vector_matches_current_name() -> None:
    """整份写入落地时版本已变。修复后的向量必须按库里的新名字重算。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="刚创建", version=0)])
    es = _FakeTagEs()
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)
    original = es.index

    async def _index_then_rename(*, index: str, id: str, document: dict, **_: object) -> None:
        await original(index=index, id=id, document=document)
        if tags.docs and tags.docs[0].get("version") == 0:
            tags.docs[0]["name"] = "已经更新的名字"
            tags.docs[0]["version"] = 1

    es.index = _index_then_rename

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            return [float(len(text))]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="刚创建", version=0))

    doc = es.docs[str(tid)]
    assert doc["name"] == "已经更新的名字"
    assert doc["version"] == 1
    assert doc["name_vector"] == [float(len("已经更新的名字"))]


@pytest.mark.asyncio
async def test_tag_repair_does_not_resurrect_deleted_tag() -> None:
    """修复写入时标签已经从库里删除，不能把索引重新建出来。"""
    from elasticsearch import NotFoundError

    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="刚创建", version=0)])
    es = _FakeTagEs()
    service = _tag_service(tags)
    service._es_search = SimpleNamespace(tag_dictionary_index="tags", _client=es)
    original = es.index

    async def _index_then_rename(*, index: str, id: str, document: dict, **_: object) -> None:
        await original(index=index, id=id, document=document)
        if tags.docs and int(tags.docs[0].get("version") or 0) == 0:
            tags.docs[0]["name"] = "已更新"
            tags.docs[0]["version"] = 1

    async def _update_deletes_tag(*, index: str, id: str, doc=None, script=None, **_: object) -> None:
        tags.docs.clear()
        es.docs.pop(id, None)
        raise NotFoundError("missing", None, None)

    es.index = _index_then_rename
    es.update = _update_deletes_tag

    class _Encoder:
        async def encode_one(self, text: str, language: str = "zh") -> list[float]:
            if text == "已更新":
                tags.docs.clear()
                es.docs.pop(str(tid), None)
            return [float(len(text))]

    service._encoder = _Encoder()
    await tag_writes._index_tag(service, str(tid), _content_tag(tid, name="刚创建", version=0))

    assert str(tid) not in es.docs
    assert tags.docs == []


@pytest.mark.asyncio
async def test_stale_tag_delete_of_live_tag_is_lease_conflict() -> None:
    """编号更小的删除碰到已被接管的标签时，标签还在，不能报成不存在。"""
    tid = ObjectId()
    tags = _Collection([_content_tag(tid, name="还在", version=2, fence_token=6)])
    service = _tag_service(tags)
    tombstones = _Collection([])
    service._db = lambda: {writes._DELETE_TOMBSTONES_COLLECTION: tombstones}

    with pytest.raises(writes.StaleLeaseError, match="接管") as raised:
        await tag_writes._delete_one_locked(service, str(tid), tid, fence=5)

    assert "不存在" not in str(raised.value)
    assert await tags.find_one({"_id": tid}) is not None
    assert tombstones.docs == []
