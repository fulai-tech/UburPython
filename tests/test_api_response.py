"""路由成功响应信封测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.core.codes import HttpStatus
from app.main import create_app
from app.schemas.audio import SearchAudioData
from tests.test_exceptions import _assert_envelope


def test_search_returns_http_200_envelope() -> None:
    mock_service = MagicMock()
    mock_service.search_audio = AsyncMock(
        return_value=SearchAudioData(
            materials=[
                {
                    "id": "6a33a7928030d4cf420efeb6",
                    "audio_name": "雨声",
                    "audio_url": "https://cdn.example.com/a.mp3",
                    "sleep_stage_tags": [{"tag_id": "s1", "code": "unwind", "name": "放松"}],
                    "content_form_tags": [{"tag_id": "c1", "code": "rain", "name": "雨声"}],
                    "mechanism_tags": [],
                    "audio_engineering_tags": [],
                    "medical_risk_tags": [],
                    "evidence_level_tags": [{"tag_id": "e1", "code": "B", "name": "中等证据"}],
                }
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
    assert len(body["data"]["materials"]) == 1
    assert body["data"]["materials"][0]["audio_name"] == "雨声"
    assert body["data"]["materials"][0]["content_form_tags"][0]["name"] == "雨声"
    assert response.status_code == body["code"]


def test_create_audio_accepts_somni_body() -> None:
    mock_service = MagicMock()
    mock_service.create_audio = AsyncMock(
        return_value={
            "id": "seed_001",
            "audio_name": "深夜雨声",
            "audio_url": "https://cdn.example.com/a.mp3",
            "sleep_stage_tags": [{"tag_id": "s1", "code": "unwind", "name": "放松"}],
        }
    )
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).post(
            "/api/audio",
            json={
                "audio_name": "深夜雨声",
                "audio_url": "https://cdn.example.com/a.mp3",
                "sleep_stage_tags": [{"tag_id": "s1", "code": "unwind", "name": "放松"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    _assert_envelope(body, HttpStatus.OK)
    assert body["data"]["id"] == "seed_001"
    assert body["data"]["audio_name"] == "深夜雨声"
    mock_service.create_audio.assert_awaited_once()


def test_update_audio_accepts_partial_somni_body() -> None:
    mock_service = MagicMock()
    mock_service.update_audio = AsyncMock()
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).put(
            "/api/audio/674a1b2c3d4e5f6789012345",
            json={"description": "仅更新描述"},
        )

    assert response.status_code == 200
    body = response.json()
    _assert_envelope(body, HttpStatus.OK)
    assert body["msg"] == "更新成功"
    mock_service.update_audio.assert_awaited_once()
