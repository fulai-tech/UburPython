"""对外 HTTP 路由（唯一 audio 路由组，4 个端点）。

对应规范 §四：/api/audio 下 POST/PUT/DELETE 与 POST /api/audio/search。
所有接口返回统一信封 { code, msg, data, timestamp }。
"""

from fastapi import APIRouter, Depends

from app.core.exceptions import ServiceNotReadyError
from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.schemas.response import ApiResponse, success
from app.services.audio import AudioService

router = APIRouter(prefix="/audio", tags=["audio"])


def get_audio_service() -> AudioService:
    """从进程单例取 AudioService；lifespan 未完成时返回 503。"""
    from app.main import get_app_state

    state = get_app_state()
    if state.audio_service is None:
        raise ServiceNotReadyError()
    return state.audio_service


@router.post("", response_model=ApiResponse, status_code=200)
async def create_audio(
    body: CreateAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> ApiResponse:
    """创建音频：comm-service 写 Mongo → EsSync 同步 ES（含 embedding）。"""
    result = await service.create_audio(body)
    return success(data=result.model_dump(), msg="创建成功")


@router.put("/{material_id}", response_model=ApiResponse)
async def update_audio(
    material_id: str,
    body: UpdateAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> ApiResponse:
    """更新音频；material_id 走路径参数，不进请求体（规范 §四）。"""
    await service.update_audio(material_id, body)
    return success(msg="更新成功")


@router.delete("/{material_id}", response_model=ApiResponse)
async def delete_audio(
    material_id: str,
    service: AudioService = Depends(get_audio_service),
) -> ApiResponse:
    """删除音频：comm 删 Mongo 真值 + EsSync 删 ES 索引副本。"""
    await service.delete_audio(material_id)
    return success(msg="删除成功")


@router.post("/search", response_model=ApiResponse)
async def search_audio(
    body: SearchAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> ApiResponse:
    """三维度检索：只读 ES，返回 somni_audio_materials 索引文档列表。"""
    result = await service.search_audio(body)
    return success(data=result.model_dump(), msg="检索成功")
