"""CommClient 单元测试（gRPC stub  mock，不连真实 comm-service）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bionode_grpc_clients.comm import AUDIO_MATERIAL_STATUS_PUBLISHED, CommClient
from app.bionode_grpc_clients.comm.grpc_gen import bionode_comm_pb2, bionode_common_pb2
from app.core.config import Settings


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
