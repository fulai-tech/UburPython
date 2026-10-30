"""Export a read-only Mongo dictionary and check/sync the DeepHealth tag contract.

Run with PYTHONPATH=. python scripts/export_audio_tag_contract.py --mirror-root PATH.
Add --refresh --env-file PATH to explicitly refresh the local dictionary from Mongo.
No database writes or embedding regeneration are performed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

from app.core.audio_tags import build_dictionary_aliases, validate_audio_tag_contract

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "app/core"


def dph_module(source: str) -> str:
    """Render the same resolver with DeepHealth's configuration-path adapter."""
    return (
        source.replace("from pathlib import Path\n", "")
        .replace(
            "from typing import Any\n",
            "from typing import Any\n\nfrom config import get_project_root\n",
        )
        .replace(
            'Path(__file__).with_name("audio_tag_aliases.json")',
            '(get_project_root() / "conf/somni/audio_tag_aliases.json")',
        )
        .replace(
            'Path(__file__).with_name("audio_tag_aliases_v2.json")',
            '(get_project_root() / "conf/somni/audio_tag_aliases_v2.json")',
        )
        .replace(
            'Path(__file__).with_name("audio_preference_rules.json")',
            '(get_project_root() / "conf/somni/audio_preference_rules.json")',
        )
    )


def synchronize(mirror: Path, *, write: bool) -> None:
    """Check assets and executable resolver logic together; formatting is immaterial."""
    asset = json.loads((CORE / "audio_tag_aliases.json").read_text())
    rules = json.loads((CORE / "audio_preference_rules.json").read_text())
    validate_audio_tag_contract(asset["dictionary"], rules)
    if asset["aliases"] != build_dictionary_aliases(asset["dictionary"]):
        raise ValueError("dictionary_alias_drift")
    outputs = {
        mirror / "conf/somni/audio_tag_aliases.json": (CORE / "audio_tag_aliases.json").read_text(),
        mirror / "conf/somni/audio_preference_rules.json": (
            CORE / "audio_preference_rules.json"
        ).read_text(),
    }
    outputs[mirror / "conf/somni/audio_tag_aliases_v2.json"] = (
        CORE / "audio_tag_aliases_v2.json"
    ).read_text()
    # Resolver policies for unmatched user prose differ between services. Check
    # the shared machine identity functions; never overwrite a service policy.
    local_tree = ast.parse(dph_module((CORE / "audio_tags.py").read_text()))
    remote_tree = ast.parse((mirror / "src/utils/audio_tags.py").read_text())
    for name in (
        "build_dictionary_aliases",
        "canonical_tag",
        "validated_audio_tags",
        "reference_tag",
    ):
        local = next(
            node
            for node in local_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        remote = next(
            node
            for node in remote_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        if ast.dump(local) != ast.dump(remote):
            raise ValueError(f"audio_identity_resolver_drift:{name}")
    for path, expected in outputs.items():
        if write:
            path.write_text(expected)
        elif path.suffix == ".py":
            if ast.dump(ast.parse(path.read_text())) != ast.dump(ast.parse(expected)):
                raise ValueError("audio_resolver_mirror_drift")
        elif json.loads(path.read_text()) != json.loads(expected):
            raise ValueError("audio_dictionary_mirror_drift")


def refresh(env_file: Path) -> None:
    """Read explicitly selected operator profile without printing connection values."""
    from dotenv import dotenv_values
    from pymongo import MongoClient

    values = dotenv_values(env_file)
    uri = (
        os.getenv("SOMNI_MONGODB_URI")
        or values.get("SOMNI_MONGODB_URI")
        or values.get("SOMNI_MONGO_URI")
    )
    if not uri:
        raise ValueError("missing_somni_mongodb_uri")
    with MongoClient(uri, serverSelectionTimeoutMS=8000) as client:
        source = client["Somni"].somni_audio_tag_dictionary.find(
            {},
            {
                "code": 1,
                "type": 1,
                "name": 1,
                "name_en": 1,
                "status": 1,
                "parent_tag_id": 1,
            },
        )
        rows = [
            {
                **{key: row.get(key, "") for key in ("code", "type", "name", "name_en", "status")},
                "id": str(row["_id"]),
                "parent_tag_id": str(row.get("parent_tag_id") or ""),
            }
            for row in source
        ]
    rows.sort(key=lambda row: (row["type"], row["code"]))
    invalid = [row for row in rows if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", row["code"])]
    rows = [row for row in rows if row not in invalid]
    rules = json.loads((CORE / "audio_preference_rules.json").read_text())
    validate_audio_tag_contract(rows, rules)
    asset = {
        "dictionary": rows,
        "aliases": build_dictionary_aliases(rows),
        "excluded_records": [{"record": row, "reason": "invalid_machine_code"} for row in invalid],
    }
    (CORE / "audio_tag_aliases.json").write_text(
        json.dumps(asset, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.refresh:
        if not args.write:
            parser.error("--refresh requires --write")
        refresh(args.env_file)
    synchronize(args.mirror_root, write=args.write)
    print("audio_tag_contract_synced")


if __name__ == "__main__":
    main()
