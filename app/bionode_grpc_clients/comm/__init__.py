"""comm-service gRPC 客户端。"""

from app.bionode_grpc_clients.comm.client import (
    AUDIO_MATERIAL_STATUS_PUBLISHED,
    CommClient,
)

__all__ = ["AUDIO_MATERIAL_STATUS_PUBLISHED", "CommClient"]
