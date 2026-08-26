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
    assert hasattr(uburnode_somni_pb2_grpc, "AudioServiceServicer")
    quiz = uburnode_somni_pb2.DESCRIPTOR.services_by_name["QuizService"]
    assert quiz.full_name == "uburnode.somni.v1.QuizService"
    assert "AnswerItem" not in uburnode_somni_pb2.DESCRIPTOR.message_types_by_name
    answers_field = uburnode_somni_pb2.GetAnswerRes.DESCRIPTOR.fields_by_name["answers"]
    assert answers_field.type == answers_field.TYPE_STRING
    assert not answers_field.is_repeated
