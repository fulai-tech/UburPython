"""应用层可预期异常（body.code 与 HTTP 状态码一致）。"""

from __future__ import annotations

from app.core.codes import HttpStatus


class AppError(Exception):
    """可预期错误；响应体 code 等于 status_code。"""

    def __init__(self, *, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)

    @property
    def code(self) -> int:
        return self.status_code


class ServiceNotReadyError(AppError):
    def __init__(self, message: str = "服务尚未就绪，请稍后重试") -> None:
        super().__init__(message=message, status_code=HttpStatus.SERVICE_UNAVAILABLE)


class CommMaterialNotFoundError(AppError):
    def __init__(self, name: str) -> None:
        super().__init__(
            message=f"创建成功但按名称未找到原料：{name}",
            status_code=HttpStatus.BAD_GATEWAY,
        )


class EncoderNotReadyError(AppError):
    def __init__(self) -> None:
        super().__init__(
            message="向量编码器未就绪，请关闭 APP_DEBUG 并确保模型已加载",
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
        )


class ElasticsearchUnavailableError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            message=f"Elasticsearch 不可用：{detail}",
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
        )


class UpstreamGrpcError(AppError):
    def __init__(self, *, grpc_code: str, message: str) -> None:
        super().__init__(
            message=message,
            status_code=_grpc_to_http_status(grpc_code),
        )
        self.grpc_code = grpc_code


class InternalError(AppError):
    def __init__(self, message: str = "服务器内部错误，请稍后重试") -> None:
        super().__init__(message=message, status_code=HttpStatus.INTERNAL_SERVER_ERROR)


def _grpc_to_http_status(grpc_code: str) -> int:
    mapping = {
        "INVALID_ARGUMENT": HttpStatus.BAD_REQUEST,
        "NOT_FOUND": HttpStatus.NOT_FOUND,
        "ALREADY_EXISTS": HttpStatus.CONFLICT,
        "FAILED_PRECONDITION": 412,
        "UNAVAILABLE": HttpStatus.SERVICE_UNAVAILABLE,
        "DEADLINE_EXCEEDED": HttpStatus.GATEWAY_TIMEOUT,
    }
    return mapping.get(grpc_code, HttpStatus.BAD_GATEWAY)
