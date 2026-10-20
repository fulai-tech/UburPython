"""Fixture inspection must reject foreign data and remain offline by default."""

import json
import sys
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest
from bson import json_util

from scripts import somni_preference_fixture_import as importer


def _package(tmp_path):
    target = "2026-10-20"
    users = [
        {"uid": "su_" + uuid5(NAMESPACE_URL, f"somni-preference-fixture/{target}/{i}").hex}
        for i in range(1, 6)
    ]
    data = {name: [] for name in importer.COLLECTIONS}
    for index, user in enumerate(users, start=1):
        data["somni_plans"].append(
            {
                "_id": f"fixture-plan-{index}",
                "uid": user["uid"],
                "date": target,
                "plan_id": "plan_" + str(uuid5(NAMESPACE_URL, user["uid"] + "/" + target)),
                "device_id": f"somni_pref_fixture_20261020_{index:02d}",
                "create_time": datetime(2026, 10, 20, tzinfo=UTC),
            }
        )
    manifest = {
        "synthetic": True,
        "target_local_date": target,
        "users": users,
        "counts": {name: len(rows) for name, rows in data.items()},
    }
    _write(tmp_path, manifest, data)
    return manifest, data


def _write(root, manifest, data):
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name, rows in data.items():
        (root / (name + ".json")).write_text(json_util.dumps(rows))


@pytest.mark.parametrize("extra_args", [[], ["--remove"]])
def test_default_inspection_never_opens_mongo(tmp_path, monkeypatch, capsys, extra_args):
    _package(tmp_path)

    def forbidden_client(*args, **kwargs):
        pytest.fail("dry-run attempted to open MongoDB")

    monkeypatch.setattr(importer, "MongoClient", forbidden_client)
    monkeypatch.setenv("SOMNI_MONGODB_URI", "mongodb://unused.example.invalid")
    monkeypatch.setattr(sys, "argv", ["fixture_import", str(tmp_path), *extra_args])
    importer.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["remove"] is bool(extra_args)
    assert summary["counts"]["somni_plans"] == 5


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("real_package", "synthetic_manifest_required"),
        ("foreign_user", "foreign_uid_in_package"),
        ("unowned_session", "unowned_session"),
        ("string_date", "mongo_extended_json_dates_required"),
        ("wrong_count", "manifest_count_mismatch"),
        ("foreign_plan", "synthetic_plan_identity_mismatch"),
        ("duplicate_id", "duplicate_id"),
    ],
)
def test_invalid_package_is_rejected_before_database_access(tmp_path, fault, reason):
    manifest, data = _package(tmp_path)
    if fault == "real_package":
        manifest["synthetic"] = False
    elif fault == "foreign_user":
        data["somni_plans"][0]["uid"] = "real-user"
    elif fault == "unowned_session":
        data["somni_sessions"] = [{"_id": "bad-session", "plan_id": "foreign-plan"}]
        manifest["counts"]["somni_sessions"] = 1
    elif fault == "string_date":
        data["somni_plans"][0]["create_time"] = "2026-10-20T00:00:00Z"
    elif fault == "wrong_count":
        manifest["counts"]["somni_plans"] = 6
    elif fault == "foreign_plan":
        data["somni_plans"][0]["plan_id"] = "foreign-plan"
    else:
        data["somni_plans"][1]["_id"] = data["somni_plans"][0]["_id"]
    _write(tmp_path, manifest, data)
    with pytest.raises(ValueError, match=reason):
        importer.load_package(tmp_path)
