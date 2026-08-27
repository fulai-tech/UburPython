"""proto gen 产物可 import。"""

from google.protobuf.struct_pb2 import Value

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


def test_get_hot_res_has_items() -> None:
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2

    res = uburnode_somni_pb2.GetHotRes(
        items=[uburnode_somni_pb2.HotKeyword(keyword="雨声", score=3)]
    )
    assert res.items[0].keyword == "雨声"
    assert res.items[0].score == 3


def test_tag_dict_item_has_id_parent_status() -> None:
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2

    item = uburnode_somni_pb2.TagDictItem(
        id="t1",
        parent_tag_id=Value(null_value=0),
        status="启用",
        type="content_form",
        code="rain",
        name="雨声",
        name_en="Rain",
    )
    assert item.id == "t1"
    assert item.parent_tag_id.WhichOneof("kind") == "null_value"
    assert item.status == "启用"


def test_audio_list_item_keeps_default_field_presence() -> None:
    item = uburnode_somni_pb2.AudioListItem(
        id="m1",
        audio_name="雨声",
        audio_url="",
        cover_url="",
        description="",
        vip=0,
    )
    assert item.HasField("audio_url")
    assert item.HasField("cover_url")
    assert item.HasField("description")
    assert item.HasField("vip")
    assert item.vip == 0
