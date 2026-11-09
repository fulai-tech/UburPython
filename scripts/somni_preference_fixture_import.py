"""Inspect/import/remove an explicitly selected five-user synthetic fixture package.

Dry-run is the default. No credentials are logged. Cleanup is limited to the
manifest's exact five UIDs and exact owned session/plan identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from bson import json_util
from pymongo import MongoClient, ReplaceOne

COLLECTIONS = (
    "somni_quiz_answers",
    "somni_user_profile_long_terms",
    "somni_user_profiles_short_terms",
    "somni_plans",
    "somni_plans_snapshots",
    "somni_sessions",
    "somni_device_operations",
)


def load_package(root: Path) -> tuple[dict, dict]:
    """Validate scope, ownership, date types and the known unique-index keys."""
    from datetime import datetime

    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("synthetic") is not True:
        raise ValueError("synthetic_manifest_required")
    target = manifest["target_local_date"]
    expected = {
        "su_" + uuid5(NAMESPACE_URL, f"somni-preference-fixture/{target}/{i}").hex
        for i in range(1, 6)
    }
    if {u["uid"] for u in manifest["users"]} != expected or len(manifest["users"]) != 5:
        raise ValueError("exact_five_generated_users_required")
    data = {name: json_util.loads((root / (name + ".json")).read_text()) for name in COLLECTIONS}
    for name, rows in data.items():
        if len(rows) != manifest["counts"][name]:
            raise ValueError("manifest_count_mismatch")
        if len({row["_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate_id")
        for row in rows:
            if name != "somni_sessions" and row.get("uid") not in expected:
                raise ValueError("foreign_uid_in_package")
            for field in ("create_time", "update_time", "started_at", "ended_at", "executed_at"):
                if field in row and not isinstance(row[field], datetime):
                    raise ValueError("mongo_extended_json_dates_required")
    expected_devices = {
        f"somni_pref_fixture_{target.replace('-', '')}_{i:02d}" for i in range(1, 6)
    }
    for row in data["somni_plans"]:
        expected_plan = "plan_" + str(uuid5(NAMESPACE_URL, row["uid"] + "/" + row["date"]))
        if row["plan_id"] != expected_plan or row.get("device_id") not in expected_devices:
            raise ValueError("synthetic_plan_identity_mismatch")
    ownership = {row["plan_id"]: row["uid"] for row in data["somni_plans"]}
    for row in data["somni_sessions"]:
        if row["plan_id"] not in ownership:
            raise ValueError("unowned_session")
    unique = {
        "somni_device_operations": [("op_id",), ("sn", "request_id")],
        "somni_plans_snapshots": [("snapshot_id",), ("request_id", "plan_id")],
        "somni_sessions": [("session_id",)],
    }
    for name, indexes in unique.items():
        for fields in indexes:
            if len({tuple(row.get(key) for key in fields) for row in data[name]}) != len(
                data[name]
            ):
                raise ValueError("unique_index_collision")
    return manifest, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--database", default="Somni")
    parser.add_argument(
        "--apply", action="store_true", help="Explicitly write to the configured test DB"
    )
    parser.add_argument(
        "--remove", action="store_true", help="Remove only the package's exact identifiers"
    )
    args = parser.parse_args()
    manifest, data = load_package(args.package)
    print(
        json.dumps(
            {
                "dry_run": not args.apply,
                "remove": args.remove,
                "database": args.database,
                "counts": {name: len(rows) for name, rows in data.items()},
                "target_date": manifest["target_local_date"],
            },
            ensure_ascii=False,
        )
    )
    if not args.apply:
        return
    uri = os.environ.get("SOMNI_MONGODB_URI")
    if not uri:
        raise ValueError("SOMNI_MONGODB_URI_required_in_environment")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=15000)
    try:
        db = client[args.database]
        # Check foreign ownership of colliding plans before any mutation.
        for row in data["somni_plans"]:
            if db.somni_plans.find_one(
                {"plan_id": row["plan_id"], "uid": {"$ne": row["uid"]}}, {"_id": 1}
            ):
                raise ValueError("existing_plan_owner_conflict")
        for name, rows in data.items():
            if args.remove:
                # Exact _ids only; never delete real users by a broad prefix.
                receipt = db[name].delete_many({"_id": {"$in": [row["_id"] for row in rows]}})
                print(name, "deleted", receipt.deleted_count)
            else:
                receipt = db[name].bulk_write(
                    [ReplaceOne({"_id": row["_id"]}, row, upsert=True) for row in rows],
                    ordered=True,
                )
                print(name, "matched", receipt.matched_count, "upserted", receipt.upserted_count)
    except Exception as exc:
        # Driver errors may include hosts and source documents. Do not echo them.
        raise SystemExit(
            f"fixture_operation_failed:{type(exc).__name__}; check test DB access and indexes"
        ) from None
    finally:
        client.close()


if __name__ == "__main__":
    main()
