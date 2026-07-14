"""CommClient Somni Create/Update 请求映射单测。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bionode_grpc_clients.comm import AUDIO_MATERIAL_STATUS_PUBLISHED, CommClient
from app.bionode_grpc_clients.comm.grpc_gen import bionode_comm_pb2, bionode_common_pb2
from app.core.config import Settings
from app.schemas.audio import CreateAudioRequest, UpdateAudioRequest


@pytest.mark.asyncio
async def test_create_audio_material_maps_somni_fields() -> None:
    client = CommClient(Settings())
    stub = MagicMock()
    stub.CreateAudioMaterial = AsyncMock(return_value=bionode_comm_pb2.EmptyRes())
    client._stub = stub

    await client.create_audio_material(
        CreateAudioRequest.model_validate(
            {
                "audio_name": "雨声",
                "audio_url": "https://cdn.example.com/a.mp3",
                "operation_type": 1,
                "created_by": "agent",
                "sleep_stage_tags": [
                    {"tag_id": "t1", "code": "unwind", "name": "放松"}
                ],
                "audio_engineering_tags": [
                    {
                        "tag_id": "e1",
                        "code": "event_density",
                        "name": "密度",
                        "value": {"tag_id": "v1", "code": "low", "name": "低"},
                        "band_values": [0.1, 0.2],
                        "relative_loudness": -3.5,
                    }
                ],
            }
        )
    )

    req = stub.CreateAudioMaterial.await_args.args[0]
    assert req.audio_name == "雨声"
    assert req.audio_url == "https://cdn.example.com/a.mp3"
    assert req.operation_type == 1
    assert req.created_by == "agent"
    assert len(req.sleep_stage_tags) == 1
    assert req.sleep_stage_tags[0].code == "unwind"
    assert len(req.audio_engineering_tags) == 1
    eng = req.audio_engineering_tags[0]
    assert eng.value.code == "low"
    assert list(eng.band_values) == [0.1, 0.2]
    assert eng.relative_loudness == pytest.approx(-3.5)


@pytest.mark.asyncio
async def test_update_audio_material_maps_partial_and_status() -> None:
    client = CommClient(Settings())
    stub = MagicMock()
    stub.UpdateAudioMaterial = AsyncMock(return_value=bionode_comm_pb2.EmptyRes())
    client._stub = stub

    await client.update_audio_material(
        "abc123",
        UpdateAudioRequest.model_validate({"description": "新描述", "status": False}),
    )

    req = stub.UpdateAudioMaterial.await_args.args[0]
    assert req.id == "abc123"
    assert req.description == "新描述"
    assert req.status == 0
    assert not req.HasField("audio_name")


@pytest.mark.asyncio
async def test_list_audio_materials_by_name_uses_published_status() -> None:
    client = CommClient(Settings())
    stub = MagicMock()
    stub.ListAudioMaterials = AsyncMock(
        return_value=bionode_comm_pb2.AudioMaterialListRes(materials=[])
    )
    client._stub = stub

    await client.list_audio_materials_by_name("测试音频")

    call_args = stub.ListAudioMaterials.await_args[0][0]
    assert call_args.name == "测试音频"
    assert call_args.status == AUDIO_MATERIAL_STATUS_PUBLISHED
    assert call_args.page.order_by == "create_time desc"


@pytest.mark.asyncio
async def test_list_audio_materials_page_uses_published_and_pagination() -> None:
    client = CommClient(Settings())
    stub = MagicMock()
    material = bionode_comm_pb2.AudioMaterialInfo(id="abc", name="海浪声白噪音")
    stub.ListAudioMaterials = AsyncMock(
        return_value=bionode_comm_pb2.AudioMaterialListRes(
            materials=[material],
            page=bionode_common_pb2.PageResponse(total=1, page=1, page_size=50),
        )
    )
    client._stub = stub

    materials, total = await client.list_audio_materials_page(page=2, page_size=50)

    assert len(materials) == 1
    assert materials[0].id == "abc"
    assert total == 1
    call_args = stub.ListAudioMaterials.await_args[0][0]
    assert call_args.status == AUDIO_MATERIAL_STATUS_PUBLISHED
    assert call_args.page.page == 2
    assert call_args.page.page_size == 50
    assert call_args.page.order_by == "update_time desc"
