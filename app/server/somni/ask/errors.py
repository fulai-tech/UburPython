"""AskContext 的可预期失败。status 使用 gRPC 状态名。"""

from __future__ import annotations


class AskContextError(Exception):
    def __init__(self, status: str, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)
