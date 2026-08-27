from app.schemas.search_material import project_search_material


def test_project_keeps_only_api_fields() -> None:
    raw = {
        "_id": "abc",
        "id": "abc",
        "audio_name": "雨声",
        "description": "轻柔雨声",
        "audio_url": "https://cdn/a.mp3",
        "cover_url": "https://cdn/a-cover.jpg",
        "status": True,
        "sleep_stage_tags": [{"tag_id": "s1", "name": "放松"}],
        "mechanism_tags": [{"tag_id": "m1", "name": "放松"}],
        "content_form_tags": [
            {
                "tag_id": "c1",
                "code": "rain",
                "name": "雨声",
                "parent_tag_id": "p1",
                "en_name": "rain",
            }
        ],
        "audio_engineering_tags": [
            {
                "tag_id": "e1",
                "code": "event_density",
                "name": "事件密度",
                "value": {"tag_id": "v1", "code": "low", "name": "低"},
                "band_values": [0.1],
            }
        ],
        "evidence_level_tags": [{"code": "B"}],
        "recommend_weight": 0.75,
        "_description_score": 0.9,
    }
    out = project_search_material(raw)
    assert out == {
        "_id": "abc",
        "audio_name": "雨声",
        "description": "轻柔雨声",
        "audio_url": "https://cdn/a.mp3",
        "cover_url": "https://cdn/a-cover.jpg",
        "content_form_tags": [{"name": "雨声", "parent_tag_id": "p1"}],
        "audio_engineering_tags": [
            {"code": "event_density", "value": {"code": "low"}}
        ],
    }


def test_project_id_fallback_to_id_field() -> None:
    out = project_search_material({"id": "x1", "audio_name": "a", "audio_url": "u"})
    assert out["_id"] == "x1"
    assert "id" not in out


def test_project_missing_value_is_null() -> None:
    out = project_search_material(
        {
            "_id": "1",
            "audio_engineering_tags": [{"code": "tempo"}],
        }
    )
    assert out["audio_engineering_tags"] == [{"code": "tempo", "value": None}]
    assert out["cover_url"] == ""


def test_project_cover_url_empty_when_absent() -> None:
    out = project_search_material(
        {"_id": "1", "audio_name": "a", "audio_url": "u"}
    )
    assert out["cover_url"] == ""
