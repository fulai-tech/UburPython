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
