"""proto gen 产物可 import。"""

from app.uburnode_grpc.grpc_gen import (
    uburnode_pb2,
    uburnode_pb2_grpc,
    uburnode_somni_pb2,
    uburnode_somni_pb2_grpc,
)


def test_handboard_package() -> None:
    assert uburnode_pb2.DESCRIPTOR.package == "uburnode.v1"
    assert hasattr(uburnode_pb2_grpc, "AudioServiceServicer")
    assert hasattr(uburnode_pb2_grpc, "QuizServiceServicer")


def test_somni_package() -> None:
    assert uburnode_somni_pb2.DESCRIPTOR.package == "uburnode.somni.v1"
    assert hasattr(uburnode_somni_pb2_grpc, "QuizServiceServicer")
    assert hasattr(uburnode_somni_pb2_grpc, "ReportServiceServicer")
    assert not hasattr(uburnode_somni_pb2_grpc, "AudioServiceServicer")
