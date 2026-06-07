"""路由成功响应信封测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.core.codes import HttpStatus
from app.main import create_app
from app.schemas.audio import AudioMaterialData, AudioResult, AudioTagsInput, EvidenceLevel, SearchAudioData
from tests.test_exceptions import _assert_envelope


def test_search_returns_http_200_envelope() -> None:
    mock_service = MagicMock()
    mock_service.search_audio = AsyncMock(
        return_value=SearchAudioData(
            audios=[
                AudioResult(
                    audio_url="https://cdn.example.com/a.mp3",
                    audio_name="雨声",
                    tags=AudioTagsInput(content_form=["雨声"]),
                    evidence_level=EvidenceLevel.B,
                    recommend_weight=0.75,
                )
            ]
        )
    )
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).post(
            "/api/audio/search",
            json={"sleep_stage_tags": ["深睡"], "content_tags": ["雨声"], "top_k": 5},
        )

    assert response.status_code == 200
    body = response.json()
    _assert_envelope(body, HttpStatus.OK)
    assert body["msg"] == "检索成功"
    assert len(body["data"]["audios"]) == 1
    assert body["data"]["audios"][0]["tags"]["content_form"] == ["雨声"]
    assert "vector_id" not in str(body["data"]["audios"][0]["tags"])
    assert response.status_code == body["code"]


def test_create_audio_accepts_six_dimension_tags() -> None:
    mock_service = MagicMock()
    mock_service.create_audio = AsyncMock(
        return_value=AudioMaterialData(
            id="seed_001",
            name="深夜雨声",
            tags=["sleep:放松", "content:雨声"],
            audio_info={"meta_data": {"url": "https://cdn.example.com/a.mp3"}},
        )
    )
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).post(
            "/api/audio",
            json={
                "category_code": 8,
                "noise_color": None,
                "level": 2,
                "name": "深夜雨声",
                "description": "",
                "tags": {
                    "sleep_stage": ["放松"],
                    "content_form": ["雨声"],
                    "mechanism": [],
                    "audio_feat": [],
                    "rhythm": [],
                    "risk_control": [],
                },
                "audio_info": {
                    "meta_data": {
                        "url": "https://cdn.example.com/a.mp3",
                        "duration_sec": 600,
                    },
                    "is_loopable": True,
                    "is_voice": False,
                },
                "evidence_level": "B",
                "recommend_weight": 0.75,
            },
        )

    assert response.status_code == 200
    body = response.json()
    _assert_envelope(body, HttpStatus.OK)
    assert body["data"]["id"] == "seed_001"
    assert body["data"]["name"] == "深夜雨声"
    assert body["data"]["audio_info"]["meta_data"]["url"] == "https://cdn.example.com/a.mp3"
    mock_service.create_audio.assert_awaited_once()
