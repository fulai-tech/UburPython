"""功能手板 audio proto ↔ Pydantic。"""

from __future__ import annotations

from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from pydantic import ValidationError

from app.core.codes import HttpStatus
from app.core.exceptions import AppError
from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioData,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.uburnode_grpc.grpc_gen import uburnode_pb2


class GrpcValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=HttpStatus.UNPROCESSABLE_ENTITY)


def create_req_to_pydantic(req: uburnode_pb2.CreateAudioReq) -> CreateAudioRequest:
    return _parse_create(_message_to_request_dict(req))


def update_req_to_pydantic(
    req: uburnode_pb2.UpdateAudioReq,
) -> tuple[str, UpdateAudioRequest]:
    if not req.material_id.strip():
        raise GrpcValidationError("material_id 不能为空")
    payload = _message_to_request_dict(req, exclude=("material_id",))
    try:
        body = UpdateAudioRequest.model_validate(payload)
    except ValidationError as exc:
        raise GrpcValidationError(_format_validation(exc)) from exc
    return req.material_id, body


def search_req_to_pydantic(req: uburnode_pb2.SearchAudioReq) -> SearchAudioRequest:
    payload: dict[str, Any] = {
        "sleep_stage_tags": list(req.sleep_stage_tags),
        "content_tags": list(req.content_tags),
        "disliked_tags": list(req.disliked_tags),
    }
    if req.HasField("query_text"):
        payload["query_text"] = req.query_text
    if req.HasField("top_k"):
        payload["top_k"] = req.top_k
    try:
        return SearchAudioRequest.model_validate(payload)
    except ValidationError as exc:
        raise GrpcValidationError(_format_validation(exc)) from exc


def dict_to_audio_material_res(data: dict[str, Any]) -> uburnode_pb2.AudioMaterialRes:
    material = uburnode_pb2.AudioMaterial()
    ParseDict(data, material, ignore_unknown_fields=True)
    return uburnode_pb2.AudioMaterialRes(material=material)


def search_data_to_res(data: SearchAudioData) -> uburnode_pb2.SearchAudioRes:
    res = uburnode_pb2.SearchAudioRes()
    for item in data.materials:
        struct = Struct()
        struct.update(item if isinstance(item, dict) else {})
        res.materials.append(struct)
    return res


def ok_operation(msg: str = "ok") -> uburnode_pb2.OperationResponse:
    return uburnode_pb2.OperationResponse(ok=True, msg=msg)


def _parse_create(payload: dict[str, Any]) -> CreateAudioRequest:
    try:
        return CreateAudioRequest.model_validate(payload)
    except ValidationError as exc:
        raise GrpcValidationError(_format_validation(exc)) from exc


def _message_to_request_dict(msg: object, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    raw = MessageToDict(msg, preserving_proto_field_name=True)  # type: ignore[arg-type]
    for key in exclude:
        raw.pop(key, None)
    return raw


def _format_validation(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg', '')}"
        for err in exc.errors()
    )
