from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from pymongo.errors import PyMongoError

from app.core.config import Settings
from app.server.somni.facts.calc import (
    build_series,
    compare_plan_execution,
    diff_plan_versions,
    finite_number,
    normalize_daily_metrics,
    normalize_metric_windows,
    normalize_somni_record_metrics,
    select_somni_record_rows,
    summarize_execution,
)
from app.server.somni.facts.models import (
    PROFILE_FIELDS,
    DataError,
    canonical_hash,
    section,
    timestamp,
    utc_now,
    validate_profile,
)
from app.server.somni.facts.profile_provenance import compact_profile_provenance
from app.server.somni.facts.sleep_debt import calculate_sleep_debt
from app.server.somni.facts.store import FactsStore
from app.server.somni.facts.time import (
    attribute_sleep_date,
    current_generation_inputs,
    current_generation_read,
    instant,
    parse_date,
    parse_offset,
    resolve_scope,
    visible_by,
)
from app.server.somni.quiz.answer_value import concrete_answer_value


def _source_ids(value: Any) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_source_ids(item) for item in value)) if value else set()
    if isinstance(value, dict):
        refs = {ref for ref in value.get("source_refs", []) if isinstance(ref, str)}
        for key, child in value.items():
            if key not in {"source_refs", "operation_bursts", "reference_table"}:
                refs.update(_source_ids(child))
        return refs
    return set()


def _pick(row: dict, fields: str) -> dict:
    return {key: row[key] for key in fields.split() if key in row and row[key] is not None}


def _concrete_questionnaire_answer(answer: dict, questions: dict[str, dict]) -> dict:
    item = _pick(
        answer,
        "question_id question_version field_code title input_type value extra_input "
        "unit scale option_codes",
    )
    question = questions.get(str(answer.get("question_id") or ""))
    options = question.get("options") if isinstance(question, dict) else None
    if "value" in answer:
        item["value"] = concrete_answer_value(
            str(answer.get("input_type") or ""),
            answer.get("value"),
            options,
            str(answer.get("extra_input") or ""),
        )
    return item


def collect_source_refs(row: dict, source: str, manifest: dict) -> list[str]:
    identifier = (
        row.get("_id")
        or row.get("reports_id")
        or row.get("snapshot_id")
        or row.get("generation_id")
        or row.get("observation_id")
        or row.get("op_id")
        or row.get("session_id")
    )
    if not identifier:
        return []
    ref = f"{source}:{identifier}"
    entry = {"ref": ref, "source_type": source, "record_id": str(identifier)}
    for key in ("revision", "content_hash"):
        if row.get(key) is not None:
            entry[key] = str(row[key])
    for source_key, target in (
        ("received_at", "received_at_utc"),
        ("occurred_at", "occurred_at_utc"),
    ):
        value = instant(row.get(source_key))
        if value:
            entry[target] = timestamp(value)
    manifest[ref] = entry
    return [ref]


def normalize_report_vitals(
    rows: list[dict],
    uid: str,
    start_date: str,
    end_date: str,
    manifest: dict,
    excluded_metric_dates: set[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[str]]:
    excluded = excluded_metric_dates or set()
    candidates: dict[tuple[str, str], list[tuple[float, str]]] = {}
    reasons: list[str] = []
    metric_fields = {
        "sleep_hr_mean_bpm": ("hr", "bpm"),
        "sleep_resp_mean_per_min": ("br", "per_min"),
    }
    for row in rows:
        if row.get("uid") != uid:
            reasons.append("sleep_report_uid_mismatch")
            continue
        local_date = row.get("record_date")
        try:
            parsed_date = parse_date(local_date).isoformat()
        except (TypeError, ValueError):
            reasons.append("sleep_report_record_date_invalid")
            continue
        if not start_date <= parsed_date < end_date:
            reasons.append("sleep_report_outside_requested_window")
            continue
        source_refs = collect_source_refs(
            {**row, "_id": row.get("reports_id") or row.get("_id")},
            "sleep_reports",
            manifest,
        )
        if not source_refs:
            reasons.append(f"sleep_report_source_id_missing:{parsed_date}")
            continue
        vitals = row.get("overnight_vitals")
        if not isinstance(vitals, dict):
            continue
        for metric_id, (report_key, unit) in metric_fields.items():
            if (metric_id, parsed_date) in excluded:
                continue
            item = vitals.get(report_key)
            if not isinstance(item, dict) or "value" not in item:
                continue
            value = item.get("value")
            if not finite_number(value) or value < 0:
                reasons.append(f"sleep_report_metric_invalid:{metric_id}:{parsed_date}")
                continue
            candidates.setdefault((metric_id, parsed_date), []).append(
                (float(value), source_refs[0])
            )

    metrics = []
    for (metric_id, local_date), values in sorted(candidates.items()):
        unique_values = {value for value, _source_ref in values}
        if len(unique_values) != 1:
            reasons.append(f"sleep_report_language_conflict:{metric_id}:{local_date}")
            continue
        report_key = metric_fields[metric_id][0]
        metrics.append(
            {
                "metric_id": metric_id,
                "unit": metric_fields[metric_id][1],
                "source": "somni_sleep_reports",
                "method": f"overnight_vitals_{report_key}_session_mean",
                "definition_version": "daily_report_projection_v1",
                "local_date": local_date,
                "value": next(iter(unique_values)),
                "quality": "valid",
                "source_refs": [min(ref for _, ref in values)],
            }
        )
    return metrics, list(dict.fromkeys(reasons))


class GenerationDataService:
    def __init__(self, client: Any, settings: Settings, store: FactsStore | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else FactsStore(client, settings)

    async def _optional(self, query: Callable[[], Awaitable[dict]]) -> dict:
        try:
            return await query()
        except DataError as exc:
            if exc.reason in ("source_time_unverified", "historical_version_unavailable"):
                return section(status="partial", reasons=[exc.reason])
            if exc.code in ("NOT_FOUND", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED"):
                raise
            return section(status="unavailable", reasons=[exc.reason])
        except (PyMongoError, TimeoutError):
            return section(status="unavailable", reasons=["source_unavailable"])
        except (ValueError, TypeError, KeyError, AttributeError):
            return section(status="partial", reasons=["malformed_source_record"])

    @staticmethod
    def _envelope(uid: str, request_id: str, read_at: datetime, data: dict, manifest: dict) -> dict:
        return {
            "schema_version": "generation_inputs.v1",
            "uid": uid,
            "request_id": request_id,
            "read_started_at_utc": timestamp(read_at),
            "generated_at_utc": timestamp(utc_now()),
            "read_consistency": "best_effort",
            "source_manifest": list(manifest.values()),
            "data": data,
            "errors": [
                {"section": name, "code": "source_unavailable", "retryable": True}
                for name, value in data.items()
                if value["status"] == "unavailable"
            ],
        }

    async def _profile(
        self,
        uid: str,
        snapshot_id: str,
        read_at: datetime,
        manifest: dict,
        end_date: str = "",
        *,
        verify_long_term_hash: bool = False,
    ) -> dict:
        short, long = await self.store.load_profile(uid, snapshot_id, read_at, end_date)
        if short is None:
            if snapshot_id:
                raise DataError("NOT_FOUND", "profile_not_found")
            return section(status="not_found", reasons=["no_prior_profile"])
        collect_source_refs(short, "profile_short", manifest)
        if long is None:
            return section(status="partial", reasons=["hash_linked_long_profile_missing"])
        if short.get("schema_version") == "0.0.1" and long.get("status") != "valid":
            return section(status="partial", reasons=["long_profile_not_valid"])
        collect_source_refs(long, "profile_long", manifest)
        current_id = short.get("profile_snapshot_id")
        legacy_id = short.get("snapshot_id")
        if current_id and legacy_id and current_id != legacy_id:
            raise DataError("FAILED_PRECONDITION", "profile_snapshot_id_conflict")
        resolved_id = current_id or legacy_id or ""
        if snapshot_id and resolved_id != snapshot_id:
            raise DataError("FAILED_PRECONDITION", "profile_snapshot_id_mismatch")
        if short.get("schema_version") == "0.0.1":
            if not resolved_id or short.get("status") != "active" or long.get("status") != "valid":
                return section(status="partial", reasons=["profile_saved_state_unverified"])
            if long.get("schema_version") != short["schema_version"]:
                return section(status="partial", reasons=["profile_schema_link_mismatch"])
        profile = _pick(short, " ".join(PROFILE_FIELDS))
        profile.update(
            profile_snapshot_id=resolved_id,
            long_term_profile=long.get("long_term_profile"),
        )
        for source_key, target in (("generated_at", "generated_at_utc"),):
            value = instant(short.get(source_key))
            if value:
                profile[target] = timestamp(value)
        if long.get("uid") != uid or long.get("content_hash") != short.get(
            "long_term_content_hash"
        ):
            raise DataError("PERMISSION_DENIED", "profile_link_mismatch")
        try:
            profile = compact_profile_provenance(profile)
            profile["content_hash"] = canonical_hash(profile)
            profile = validate_profile(profile, uid, verify_long_term_hash=verify_long_term_hash)
        except ValueError as exc:
            raise DataError("RESOURCE_EXHAUSTED", "profile_provenance_budget_exceeded") from exc
        except DataError as exc:
            if exc.code == "PERMISSION_DENIED":
                raise
            return section(status="partial", reasons=[exc.reason])
        return section(
            profile,
            reasons=[
                "long_term_hash_verification_skipped",
                *short.get("_profile_read_missing_reasons", []),
            ],
        )

    async def _blueprint(
        self, uid: str, generation_id: str, read_at: datetime, manifest: dict
    ) -> tuple[dict, dict | None]:
        if not generation_id:
            return section(status="not_found", reasons=["first_generation"]), None
        row = await self.store.load_blueprint(uid, generation_id, read_at)
        if row is None:
            raise DataError("NOT_FOUND", "generation_not_found")
        if row.get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "blueprint_owner_mismatch")
        if not isinstance(row.get("raw_plan"), dict):
            return section(status="partial", reasons=["raw_blueprint_missing"]), row
        if row["raw_plan"].get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "blueprint_owner_mismatch")
        if not isinstance(row.get("decision_record"), dict) or not isinstance(
            row.get("output_plans"), list
        ):
            return section(status="partial", reasons=["incomplete_blueprint"]), row
        value = _pick(row, "generation_id uid raw_plan decision_record profile_ref output_plans")
        reasons = []
        reasons.append("blueprint_hash_policy_unverified")
        when = instant(row.get("generated_at"))
        if when:
            value["generated_at_utc"] = timestamp(when)
        else:
            reasons.append("generation_time_missing")
        value["output_plans"] = [
            _pick(item, "plan_id version snapshot_id content_hash date")
            for item in row["output_plans"]
        ]
        value["source_refs"] = collect_source_refs(row, "generations", manifest)
        return section(value, "partial" if reasons else "ok", reasons), row

    async def _previous_week_blueprint(
        self,
        uid: str,
        target_local_date: str,
        read_at: datetime,
        manifest: dict,
    ) -> dict:
        """Resolve the immediately preceding effective weekly blueprint."""
        target_date = parse_date(target_local_date)
        previous_date = (target_date - timedelta(days=1)).isoformat()
        latest_date = await self.store.load_latest_active_plan_date_before(
            uid, target_local_date, read_at
        )
        if latest_date is None:
            if await self.store.has_plan_history_before(uid, target_local_date, read_at):
                raise DataError("FAILED_PRECONDITION", "previous_week_blueprint_missing")
            return section(status="not_found", reasons=["first_generation"])
        if latest_date != previous_date:
            raise DataError("FAILED_PRECONDITION", "previous_week_plan_gap")

        previous_plan_section = await self._target_plan(
            uid,
            previous_date,
            read_at,
            manifest,
        )
        if previous_plan_section["status"] != "ok":
            raise DataError("FAILED_PRECONDITION", "previous_week_plan_unavailable")
        generation_id = str(previous_plan_section["value"].get("weekly_generation_id") or "")
        if not generation_id:
            raise DataError("FAILED_PRECONDITION", "previous_week_generation_missing")
        blueprint_section, generation = await self._blueprint(uid, generation_id, read_at, manifest)
        if generation is None or blueprint_section["status"] not in ("ok", "partial"):
            raise DataError("FAILED_PRECONDITION", "previous_week_blueprint_unavailable")
        output_plans = generation.get("output_plans")
        if not isinstance(output_plans, list):
            raise DataError("FAILED_PRECONDITION", "previous_week_outputs_missing")
        try:
            output_dates = sorted(
                parse_date(item["date"])
                for item in output_plans
                if isinstance(item, dict) and isinstance(item.get("date"), str)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError("FAILED_PRECONDITION", "previous_week_output_dates_invalid") from exc
        expected_dates = [target_date - timedelta(days=offset) for offset in range(7, 0, -1)]
        if len(output_dates) != 7 or output_dates != expected_dates:
            raise DataError("FAILED_PRECONDITION", "previous_week_outputs_incomplete")
        return blueprint_section

    @current_generation_inputs
    async def get_plan_inputs(
        self,
        uid: str,
        request_id: str,
        *,
        profile_snapshot_id: str = "",
        inline_profile: dict | None = None,
        previous_generation_id: str = "",
        target_local_date: str = "",
        feedback_record_id: str = "",
        scope: str = "",
        timezone_offset: str = "",
        trigger_type: str = "",
        event_id: str = "",
        session_id: str = "",
    ) -> dict:
        if scope and scope not in ("weekly", "daily", "realtime"):
            raise DataError("INVALID_ARGUMENT", "invalid_plan_scope")
        if scope and not target_local_date:
            raise DataError("INVALID_ARGUMENT", "target_date_required")
        if scope:
            try:
                parse_date(target_local_date)
                if not timezone_offset:
                    raise ValueError("timezone_required")
                parse_offset(timezone_offset)
            except ValueError as exc:
                raise DataError("INVALID_ARGUMENT", "invalid_plan_temporal_scope") from exc
            if profile_snapshot_id or inline_profile is not None or previous_generation_id:
                raise DataError("INVALID_ARGUMENT", "caller_selected_plan_input_not_allowed")
            if feedback_record_id:
                raise DataError("INVALID_ARGUMENT", "inline_feedback_required_for_scoped_plan")
        if scope == "realtime" and trigger_type not in (
            "pre_unwind_health",
            "unwind_fusion",
        ):
            raise DataError("INVALID_ARGUMENT", "invalid_realtime_trigger")
        if scope == "realtime" and not event_id:
            raise DataError("INVALID_ARGUMENT", "realtime_context_required")
        if scope == "realtime" and trigger_type == "unwind_fusion" and not session_id:
            raise DataError("INVALID_ARGUMENT", "realtime_session_required")
        if scope != "realtime" and (trigger_type or event_id or session_id):
            raise DataError("INVALID_ARGUMENT", "realtime_context_not_allowed")
        read_started_at = utc_now()
        read_at = read_started_at
        has_profile_input = bool(profile_snapshot_id) or inline_profile is not None
        if bool(profile_snapshot_id) == (inline_profile is not None) and has_profile_input:
            raise DataError("INVALID_ARGUMENT", "exactly_one_profile_input_required")
        if feedback_record_id and not target_local_date:
            raise DataError("INVALID_ARGUMENT", "feedback_target_date_required")
        if not has_profile_input and not target_local_date:
            raise DataError("INVALID_ARGUMENT", "target_date_required")
        if feedback_record_id != feedback_record_id.strip():
            raise DataError("INVALID_ARGUMENT", "invalid_feedback_record_id")
        manifest = {}
        if not has_profile_input and target_local_date:
            profile = await self._optional(
                lambda: self._profile(uid, "", read_at, manifest, target_local_date),
            )
        elif not has_profile_input:
            profile = section(status="not_found", reasons=["profile_input_not_requested"])
        else:
            profile = (
                section(validate_profile(inline_profile, uid))
                if inline_profile is not None
                else await self._profile(uid, profile_snapshot_id, read_at, manifest)
            )
        if scope == "weekly" and not previous_generation_id:
            blueprint = await self._previous_week_blueprint(
                uid,
                target_local_date,
                read_at,
                manifest,
            )
        else:
            blueprint, _row = await self._blueprint(uid, previous_generation_id, read_at, manifest)
        data = {"profile": profile, "previous_blueprint": blueprint}
        if target_local_date and scope != "weekly":
            target_plan = await self._target_plan(
                uid,
                target_local_date,
                read_at,
                manifest,
                stored_compat=scope == "daily",
            )
            data["target_plan"] = target_plan
            if not previous_generation_id and target_plan.get("value", {}).get(
                "weekly_generation_id"
            ):
                blueprint, _row = await self._blueprint(
                    uid,
                    target_plan["value"]["weekly_generation_id"],
                    read_at,
                    manifest,
                )
                data["previous_blueprint"] = blueprint
        if scope == "daily":
            data["preference_evidence"] = await self._optional(
                lambda: self._preference_evidence(
                    uid, target_local_date, timezone_offset, read_at, manifest
                )
            )
        if scope == "realtime":
            if session_id:
                data["execution_state"] = await self._runtime_execution_state(
                    uid,
                    session_id,
                    data.get("target_plan", section()),
                    read_at,
                    manifest,
                    trigger_type=trigger_type,
                    event_id=event_id,
                )
            else:
                data["execution_state"] = section(
                    status="partial", reasons=["session_start_state_unverified"]
                )
        if feedback_record_id:
            data["feedback"] = await self._feedback_observation(
                uid,
                feedback_record_id,
                target_local_date,
                read_at,
                manifest,
            )
        result = self._envelope(uid, request_id, read_at, data, manifest)
        result["read_started_at_utc"] = timestamp(read_started_at)
        return result

    async def _preference_evidence(
        self,
        uid: str,
        end_local_date: str,
        timezone_offset: str,
        read_at: datetime,
        manifest: dict,
    ) -> dict:
        from app.server.somni.facts.preference import aggregate_preference_evidence

        sources = await self.store.load_preference_sources(uid, end_local_date, timezone_offset)
        value = aggregate_preference_evidence(
            uid,
            end_local_date,
            timezone_offset,
            **sources,
            read_at=read_at,
        )
        # Register only evidence that survived aggregation, never all scanned operations.
        used = _source_ids(value)
        for name, rows in sources.items():
            for row in rows:
                local = {}
                collect_source_refs(row, name, local)
                for entry in local.values():
                    aliases = {entry["ref"]}
                    aliases.update(
                        f"{name}:{row[key]}"
                        for key in ("op_id", "snapshot_id", "session_id")
                        if row.get(key)
                    )
                    for ref in aliases & used:
                        manifest[ref] = {**entry, "ref": ref}
        value["source_summary"] = {
            "input_record_count": sum(len(rows) for rows in sources.values()),
            "effective_observation_count": len(value["observations"]),
            "window_start": value["start_local_date"],
            "window_end": value["review_end_local_date"],
        }
        # Bursts are intermediate diagnostics; policy uses observations/exposures only.
        bursts = value.pop("operation_bursts", [])
        value["source_summary"]["operation_burst_count"] = len(bursts)
        value["source_summary"]["operation_burst_digest"] = canonical_hash(bursts)
        reasons = value["missing_reasons"]
        return section(value, status="partial" if reasons else "ok", reasons=reasons)

    async def _feedback_observation(
        self,
        uid: str,
        observation_id: str,
        target_local_date: str,
        read_at: datetime,
        manifest: dict,
    ) -> dict:
        try:
            target_date = parse_date(target_local_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise DataError("INVALID_ARGUMENT", "invalid_target_local_date") from exc
        row = await self.store.load_feedback_observation(uid, observation_id, read_at)
        if row is None:
            return section(status="not_found", reasons=["feedback_record_not_found"])
        if row.get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "feedback_record_owner_mismatch")
        dates = row.get("target_dates")
        dates = dates if isinstance(dates, list) else []
        observation_date = row.get("local_date") or row.get("record_date")
        if observation_date != target_date and target_date not in dates:
            raise DataError("PERMISSION_DENIED", "feedback_record_date_mismatch")
        value = _pick(
            row,
            "observation_id kind field_code value unit scale local_date target_dates "
            "episode_ref plan_refs quality",
        )
        if not value.get("local_date") and observation_date:
            value["local_date"] = observation_date
        for key in ("occurred_at", "received_at", "valid_from", "valid_to"):
            when = instant(row.get(key))
            if when:
                value[key + "_utc"] = timestamp(when)
        value["source_refs"] = collect_source_refs(row, "observations", manifest)
        return section([value])

    async def _target_plan(
        self,
        uid: str,
        target_local_date: str,
        read_at: datetime,
        manifest: dict,
        *,
        stored_compat: bool = False,
    ) -> dict:
        try:
            target_date = parse_date(target_local_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise DataError("INVALID_ARGUMENT", "invalid_target_local_date") from exc
        if stored_compat:
            # Daily adjustment selects the latest user plan, never a device projection.
            rows = await self.store.load_latest_user_plan(uid, target_date, read_at)
        else:
            rows = await self.store.load_active_plan(uid, target_date, read_at)
        if not rows:
            return section(status="not_found", reasons=["active_plan_not_found"])
        if len(rows) != 1:
            raise DataError("FAILED_PRECONDITION", "active_plan_ambiguous")
        pointer = rows[0]
        if pointer.get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "active_plan_owner_mismatch")
        if pointer.get("date") != target_date:
            raise DataError("PERMISSION_DENIED", "active_plan_date_mismatch")
        plan_id = str(pointer.get("plan_id") or "")
        weekly_generation_id = str(pointer.get("weekly_generation_id") or "")
        if (
            "weekly_generation_id" not in pointer
            and "generation_id" not in pointer
            and "phases" in pointer
        ):
            return await self._stored_target_plan(uid, target_date, pointer, read_at, manifest)
        version = pointer.get("version")
        if (
            not plan_id
            or not weekly_generation_id
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
        ):
            raise DataError("FAILED_PRECONDITION", "active_plan_link_incomplete")
        blueprint_section, generation = await self._blueprint(
            uid, weekly_generation_id, read_at, manifest
        )
        if generation is None or blueprint_section["status"] not in ("ok", "partial"):
            raise DataError("FAILED_PRECONDITION", "active_plan_generation_unavailable")
        if generation.get("status") != "committed":
            raise DataError("FAILED_PRECONDITION", "active_plan_generation_not_committed")
        if generation.get("generation_id") != weekly_generation_id:
            raise DataError("PERMISSION_DENIED", "active_plan_generation_mismatch")
        outputs = generation.get("output_plans")
        if not isinstance(outputs, list) or not any(
            item.get("plan_id") == plan_id and item.get("date") == target_date
            for item in outputs
            if isinstance(item, dict)
        ):
            raise DataError("PERMISSION_DENIED", "active_plan_generation_mismatch")
        revision = await self.store.load_plan_revision(uid, plan_id, version, read_at)
        if revision is None:
            raise DataError("NOT_FOUND", "active_plan_snapshot_not_found")
        if revision.get("uid") != uid or revision.get("plan_id") != plan_id:
            raise DataError("PERMISSION_DENIED", "active_plan_snapshot_owner_mismatch")
        if type(revision.get("version")) is not int or revision["version"] != version:
            raise DataError("FAILED_PRECONDITION", "active_plan_snapshot_version_mismatch")
        lifecycle = revision.get("lifecycle")
        if isinstance(lifecycle, dict):
            lifecycle = lifecycle.get("status")
        if lifecycle is None:
            lifecycle = revision.get("status")
        if lifecycle != "active":
            raise DataError("FAILED_PRECONDITION", "active_plan_snapshot_not_active")
        if revision.get("plan_start_local_date") != target_date:
            raise DataError("PERMISSION_DENIED", "active_plan_date_mismatch")
        refs = collect_source_refs(pointer, "plans", manifest)
        refs.extend(collect_source_refs(revision, "snapshots", manifest))
        value = _pick(
            revision,
            "plan_id version snapshot_id content_hash base_plan_version edit_source phases",
        )
        value["lifecycle"] = lifecycle
        value.update(
            weekly_generation_id=weekly_generation_id,
            generation_id=str(pointer.get("generation_id") or weekly_generation_id),
            date=target_date,
            source_refs=refs,
        )
        created_at = instant(revision.get("created_at"))
        if created_at:
            value["created_at_utc"] = timestamp(created_at)
        return section(value)

    async def _stored_target_plan(
        self, uid: str, target_date: str, pointer: dict, read_at: datetime, manifest: dict
    ) -> dict:
        """Resolve legacy current plans by exact immutable revision and body hash."""
        from app.server.somni.facts.preference_storage import (
            normalize_snapshot,
            stored_phase_content,
        )

        version = pointer.get("version")
        if type(version) is not int or version < 1 or not pointer.get("plan_id"):
            raise DataError("FAILED_PRECONDITION", "active_plan_link_incomplete")
        revision = await self.store.load_plan_revision(uid, pointer["plan_id"], version, read_at)
        if revision is None:
            raise DataError("NOT_FOUND", "active_plan_snapshot_not_found")
        if revision.get("uid") != uid or revision.get("plan_id") != pointer["plan_id"]:
            raise DataError("PERMISSION_DENIED", "active_plan_snapshot_owner_mismatch")
        if type(revision.get("version")) is not int or revision["version"] != version:
            raise DataError("FAILED_PRECONDITION", "active_plan_snapshot_version_mismatch")
        normalized = normalize_snapshot(revision)
        if normalized.get("plan_start_local_date") != target_date:
            raise DataError("PERMISSION_DENIED", "active_plan_date_mismatch")
        if normalized.get("lifecycle") != "committed":
            raise DataError("FAILED_PRECONDITION", "active_plan_snapshot_not_active")
        if not revision.get("snapshot_id") or not pointer.get("phases"):
            raise DataError("FAILED_PRECONDITION", "active_plan_link_incomplete")
        if canonical_hash(stored_phase_content(pointer["phases"])) != canonical_hash(
            stored_phase_content(revision.get("phases"))
        ):
            raise DataError("FAILED_PRECONDITION", "active_plan_body_mismatch")
        refs = collect_source_refs(pointer, "plans", manifest)
        refs.extend(collect_source_refs(revision, "snapshots", manifest))
        value = _pick(revision, "plan_id version snapshot_id phases")
        value.update(
            date=target_date,
            lifecycle="active",
            edit_source=normalized.get("edit_source"),
            content_hash=canonical_hash(
                {
                    "uid": uid,
                    "plan_id": pointer["plan_id"],
                    "date": target_date,
                    "version": version,
                    "phases": stored_phase_content(revision["phases"]),
                }
            ),
            source_refs=refs,
        )
        if revision.get("trigger_reason") == "daily_preference":
            value["edit_source"] = "daily_preference"
        return section(value)

    async def _runtime_execution_state(
        self,
        uid: str,
        session_id: str,
        target_plan_section: dict,
        read_at: datetime,
        manifest: dict,
        *,
        trigger_type: str,
        event_id: str,
    ) -> dict:
        """Validate session evidence without claiming an authoritative application cursor."""
        if target_plan_section.get("status") != "ok":
            return section(
                status="unavailable", reasons=["target_plan_unavailable_for_execution_state"]
            )
        target_plan = target_plan_section.get("value") or {}
        sessions = await self.store.list_sessions(uid, read_at, session_id=session_id)
        if not sessions:
            return section(status="not_found", reasons=["sleep_session_not_found"])
        if len(sessions) != 1:
            raise DataError("FAILED_PRECONDITION", "sleep_session_ambiguous")
        session = sessions[0]
        if session.get("uid") != uid or session.get("session_id") != session_id:
            raise DataError("PERMISSION_DENIED", "sleep_session_owner_mismatch")
        if session.get("plan_id") != target_plan.get("plan_id"):
            return section(status="unavailable", reasons=["session_target_plan_mismatch"])
        if session.get("plan_version") != target_plan.get("version"):
            return section(status="unavailable", reasons=["session_target_version_mismatch"])
        if str(session.get("state") or "").lower() != "running" or session.get("ended_at"):
            return section(status="not_found", reasons=["sleep_session_not_running"])

        phase = session.get("phase")
        node_index = session.get("node_index")
        next_node_key = session.get("next_unexecuted_node_key")
        if (
            phase != "Unwind"
            or isinstance(node_index, bool)
            or not isinstance(node_index, int)
            or node_index < 0
            or not isinstance(next_node_key, str)
            or not next_node_key.strip()
        ):
            return section(status="partial", reasons=["execution_apply_boundary_unverified"])

        phases = target_plan.get("phases")
        current_phase = (
            next(
                (item for item in phases if isinstance(item, dict) and item.get("phase") == phase),
                None,
            )
            if isinstance(phases, list)
            else None
        )
        nodes = current_phase.get("nodes") if isinstance(current_phase, dict) else None
        if not isinstance(nodes, list) or not any(
            isinstance(node, dict) and node.get("node_key") == next_node_key for node in nodes
        ):
            return section(status="partial", reasons=["next_unexecuted_node_unverified"])

        events = await self.store.list_execution_events(uid, [session], read_at)
        event_rows = events.get("events", []) if isinstance(events, dict) else []
        if trigger_type == "unwind_fusion" and not any(
            isinstance(row, dict) and str(row.get("event_id") or "") == event_id
            for row in event_rows
        ):
            return section(status="not_found", reasons=["realtime_event_not_in_session"])

        collect_source_refs(session, "sessions", manifest)
        return section(status="partial", reasons=["authoritative_execution_boundary_unavailable"])

    async def _questionnaire_questions(self, rows: Any) -> dict[str, dict]:
        if not isinstance(rows, list):
            return {}
        question_ids = [
            str(answer.get("question_id"))
            for row in rows
            if isinstance(row, dict)
            for answer in (row.get("answers") or [])
            if isinstance(answer, dict) and answer.get("question_id")
        ]
        if not question_ids:
            return {}
        loaded = await self.store.load_quiz_questions(question_ids)
        return loaded if isinstance(loaded, dict) else {}

    async def _questionnaires(
        self, uid: str, read_at: datetime, manifest: dict, answer_id: str = ""
    ) -> dict:
        rows = await self.store.load_questionnaires(uid, read_at, answer_id)
        if answer_id and not rows:
            return section(status="not_found", reasons=["questionnaire_answer_not_found"])
        questions = await self._questionnaire_questions(rows)
        values, reasons = [], []
        for row in rows:
            item = _pick(row, "answer_id survey_code survey_version")
            if not all(item.get(key) for key in ("answer_id", "survey_code")):
                reasons.append("questionnaire_identity_missing")
                continue
            for source, target in (
                ("completed_at", "completed_at_utc"),
                ("received_at", "received_at_utc"),
            ):
                when = instant(row.get(source))
                if when:
                    item[target] = timestamp(when)
            item["answers"] = [
                _concrete_questionnaire_answer(answer, questions)
                for answer in row.get("answers", [])
                if isinstance(answer, dict)
            ]
            item["scores"] = [
                {
                    **_pick(score, "instrument_id value unit"),
                    "version": score.get("rule_version", ""),
                }
                for score in row.get("results", [])
            ]
            item["source_refs"] = collect_source_refs(row, "questionnaires", manifest)
            values.append(item)
        return section(values, "partial" if reasons else "ok" if values else "not_found", reasons)

    async def _observations(self, uid: str, scope: Any, read_at: datetime, manifest: dict) -> dict:
        rows = await self.store.list_observations(
            uid, scope.start(28).isoformat(), scope.end.isoformat(), read_at
        )
        superseded = {row.get("supersedes_id") for row in rows}
        values = []
        for row in rows:
            if row.get("observation_id") in superseded:
                continue
            item = _pick(
                row,
                "observation_id kind field_code value unit scale local_date target_dates "
                "episode_ref plan_refs supersedes_id quality",
            )
            for key in ("occurred_at", "received_at", "valid_from", "valid_to"):
                when = instant(row.get(key))
                if when:
                    item[key + "_utc"] = timestamp(when)
            item["source_refs"] = collect_source_refs(row, "observations", manifest)
            values.append(item)
        return section(values, "partial", ["legacy_subjective_rating_adapter_pending"])

    async def build_facts(self, uid: str, scope: Any, read_at: datetime, manifest: dict) -> dict:
        report_reasons = []
        rows = await self.store.list_sleep_episodes(
            uid,
            scope.start(28).isoformat(),
            scope.end.isoformat(),
            read_at,
            missing_reasons=report_reasons,
        )
        activity = await self.store.load_daily_metrics(
            uid,
            scope.start(28).isoformat(),
            scope.end.isoformat(),
            read_at,
            missing_reasons=report_reasons,
        )
        try:
            reports = await self.store.list_daily_sleep_reports(
                uid,
                scope.start(28).isoformat(),
                scope.end.isoformat(),
                read_at,
                missing_reasons=report_reasons,
            )
        except DataError as exc:
            reports = []
            report_reasons.append(f"sleep_reports_{exc.reason}")
        except (PyMongoError, TimeoutError):
            reports = []
            report_reasons.append("sleep_reports_unavailable")
        episodes, reasons = (
            [],
            [
                "raw_signal_and_event_aggregation_pending",
                "read_visibility_best_effort",
                "partial_metric_mapping",
            ],
        )
        reasons.extend(report_reasons)
        selected_records, selected_reasons = select_somni_record_rows(
            [row for row in rows if row.get("source_type") == "records"]
        )
        reasons.extend(selected_reasons)
        rows = [row for row in rows if row.get("source_type") != "records"] + selected_records
        for row in rows + activity:
            row["source_refs"] = collect_source_refs(
                row, row.get("source_type", "wearable_daily"), manifest
            )
        for row in rows:
            sleep = row.get("sleep") or {}
            start, end = instant(sleep.get("start_at")), instant(sleep.get("end_at"))
            if (
                not row.get("episode_id")
                or not start
                or not end
                or start >= end
                or not visible_by(end, read_at)
                or row.get("main_sleep") is not True
                or not row.get("selection_policy_version")
            ):
                reasons.append("episode_selection_or_interval_unverified")
                continue
            episode = {
                "episode_id": row["episode_id"],
                "source": row.get("source", ""),
                "sleep_date_local": row["record_date"],
                "start_at_utc": timestamp(start),
                "end_at_utc": timestamp(end),
                "main_sleep": True,
                "selection_policy_version": row["selection_policy_version"],
                "source_refs": row["source_refs"],
            }
            if row.get("revision") is not None:
                episode["revision"] = str(row["revision"])
            if row.get("timezone_offset"):
                episode["timezone_offset"] = row["timezone_offset"]
            episodes.append(episode)
        record_candidates = [row for row in rows if row.get("source_type") == "records"]
        record_rows, record_selection_reasons = select_somni_record_rows(record_candidates)
        reasons.extend(record_selection_reasons)
        wearable_rows = [row for row in rows if row.get("source_type") != "records"]
        wearable_metrics = normalize_daily_metrics(wearable_rows) + normalize_daily_metrics(
            activity, activity=True
        )
        record_metrics = normalize_daily_metrics(record_rows)
        legacy_record_metrics, record_metric_reasons = normalize_somni_record_metrics(
            record_rows, timezone_offset=scope.offset, read_at=read_at
        )
        reasons.extend(record_metric_reasons)
        legacy_record_metric_dates = {
            (item["metric_id"], item["local_date"]) for item in legacy_record_metrics
        }
        record_metrics = [
            item
            for item in record_metrics
            if item.get("quality") != "missing"
            or (item["metric_id"], item["local_date"]) not in legacy_record_metric_dates
        ]
        record_metric_dates = {
            (item["metric_id"], item["local_date"])
            for item in record_metrics
            if item.get("quality") == "valid"
        }
        record_metrics.extend(
            item
            for item in legacy_record_metrics
            if (item["metric_id"], item["local_date"]) not in record_metric_dates
        )
        preferred_metric_dates = {
            (item["metric_id"], item["local_date"])
            for item in wearable_metrics
            if item.get("quality") == "valid"
        }
        report_metrics, report_reasons = normalize_report_vitals(
            reports,
            uid,
            scope.start(28).isoformat(),
            scope.end.isoformat(),
            manifest,
            excluded_metric_dates=preferred_metric_dates,
        )
        report_metric_dates = {(item["metric_id"], item["local_date"]) for item in report_metrics}
        wearable_metrics = [
            item
            for item in wearable_metrics
            if item.get("quality") != "missing"
            or (item["metric_id"], item["local_date"]) not in report_metric_dates
        ]
        record_metrics = [
            item
            for item in record_metrics
            if (item["metric_id"], item["local_date"])
            not in preferred_metric_dates | report_metric_dates
        ]
        metrics = [*wearable_metrics, *report_metrics, *record_metrics]
        reasons.extend(report_reasons)
        if report_metrics:
            reasons.append("sleep_report_quality_and_visibility_unverified")
        windows, window_reasons = normalize_metric_windows(rows + activity, read_at)
        reasons.extend(window_reasons)
        if not windows:
            reasons.append("presleep_metric_windows_unavailable")
        value = {
            "episodes": episodes,
            "series": build_series(metrics, scope),
            "metric_windows": windows,
        }
        if episodes:
            latest = max(episodes, key=lambda row: row["end_at_utc"])
            value.update(
                latest_episode=latest,
                latest_age_days=(scope.end - parse_date(latest["sleep_date_local"])).days,
                latest_values=[
                    item
                    for item in metrics
                    if item.get("episode_id") == latest["episode_id"]
                    and item["source"] == latest["source"]
                ],
            )
        return section(value, "partial", reasons)

    @current_generation_inputs
    async def get_profile_inputs(
        self,
        uid: str,
        request_id: str,
        *,
        purpose: str,
        review_end_local_date: str,
        timezone_offset: str,
        previous_profile_snapshot_id: str = "",
        review_generation_id: str = "",
        requested_sections: list[str] | None = None,
        answer_id: str = "",
    ) -> dict:
        if purpose not in ("initial", "weekly", "correction", "plan_generation", "profile_refresh"):
            raise DataError("INVALID_ARGUMENT", "invalid_purpose")
        if purpose == "weekly" and not review_generation_id:
            raise DataError("INVALID_ARGUMENT", "weekly_generation_required")
        if purpose == "profile_refresh" and (previous_profile_snapshot_id or review_generation_id):
            raise DataError("INVALID_ARGUMENT", "profile_refresh_selectors_not_allowed")
        scope = resolve_scope(review_end_local_date, timezone_offset)
        profile_end_date = (
            (scope.end - timedelta(days=1)).isoformat()
            if purpose == "profile_refresh"
            else review_end_local_date
        )
        read_at, manifest = utc_now(), {}
        allowed_sections = {
            "questionnaires",
            "prior_profile",
            "facts",
            "observations",
            "plan_comparisons",
            "execution",
            "sleep_debt",
            "preference_evidence",
        }
        if requested_sections and not set(requested_sections) <= allowed_sections:
            raise DataError("INVALID_ARGUMENT", "unknown_requested_section")
        if purpose == "initial":
            initial_sections = {"questionnaires", "prior_profile", "facts", "sleep_debt"}
            if requested_sections and not set(requested_sections) <= initial_sections:
                raise DataError("INVALID_ARGUMENT", "initial_section_not_allowed")
            requested_sections = requested_sections or sorted(initial_sections)
        tasks = {
            "sleep_debt": lambda: self.build_sleep_debt(
                uid, (scope.end - timedelta(days=1)).isoformat(), read_at, manifest
            ),
            "questionnaires": lambda: self._questionnaires(uid, read_at, manifest, answer_id),
            "prior_profile": lambda: self._profile(
                uid,
                previous_profile_snapshot_id,
                read_at,
                manifest,
                profile_end_date,
                verify_long_term_hash=False,
            ),
            "facts": lambda: self.build_facts(uid, scope, read_at, manifest),
            "observations": lambda: self._observations(uid, scope, read_at, manifest),
            "plan_comparisons": lambda: self.build_plan_comparisons(
                uid, review_generation_id, read_at, manifest
            ),
            "execution": lambda: self._execution(uid, scope, read_at, manifest),
        }
        if purpose in {"weekly", "profile_refresh", "plan_generation"} or (
            requested_sections and "preference_evidence" in requested_sections
        ):
            tasks["preference_evidence"] = lambda: self._preference_evidence(
                uid, (scope.end - timedelta(days=1)).isoformat(), timezone_offset, read_at, manifest
            )
        if requested_sections:
            tasks = {key: value for key, value in tasks.items() if key in requested_sections}
        results = await asyncio.gather(*(self._optional(query) for query in tasks.values()))
        data = dict(zip(tasks, results, strict=True))
        used = _source_ids(data)
        # profile refs are envelope-level identity proofs, not recursive inherited sources.
        manifest = {
            ref: entry
            for ref, entry in manifest.items()
            if ref in used or entry.get("source_type") in {"profile_short", "profile_long"}
        }
        response = self._envelope(uid, request_id, read_at, data, manifest)
        response["time_context"] = {
            "review_end_local_date": review_end_local_date,
            "timezone_offset": timezone_offset,
            "time_policy_version": "fixed_offset.v1",
        }
        for days in (7, 28):
            start, end = scope.bounds(days)
            response["time_context"][f"window_{days}d"] = {
                "start_at_utc": timestamp(start),
                "end_at_utc": timestamp(end),
            }
        return response

    async def build_sleep_debt(
        self, uid: str, record_date: str, cutoff: datetime, manifest: dict
    ) -> dict:
        rows = await self.store.load_sleep_debt_rows(uid, record_date, cutoff)
        for row in rows:
            row["source_refs"] = collect_source_refs(row, "wearable_daily", manifest)
        return calculate_sleep_debt(
            rows, record_date, None if current_generation_read() else cutoff
        )

    async def _reminder_schedules(
        self, uid: str, target: str, offset: str, read_at: datetime, manifest: dict
    ) -> dict:
        previous = (parse_date(target) - timedelta(days=1)).isoformat()
        scope = resolve_scope((parse_date(target) + timedelta(days=1)).isoformat(), offset)
        lower, upper = scope.bounds(1)
        rows = await self.store.load_reminder_schedules(
            uid, [previous, target], read_at, lower, upper
        )
        candidates = {}
        reasons = [
            "calendar_sync_coverage_unverified",
            "multi_day_and_recurrence_coverage_unverified",
        ]
        for row in rows:
            identity = row.get("event_id") or str(row.get("_id", ""))
            if not identity:
                reasons.append("schedule_identity_missing")
                continue
            candidates.setdefault(identity, []).append(row)
        values = []
        for identity, versions in candidates.items():
            if len(versions) > 1:
                if not all(isinstance(row.get("revision"), int) for row in versions):
                    reasons.append("schedule_revision_ambiguous")
                    continue
                highest = max(row["revision"] for row in versions)
                versions = [row for row in versions if row["revision"] == highest]
                if len(versions) != 1:
                    reasons.append("schedule_revision_ambiguous")
                    continue
            row = versions[0]
            if row.get("status") in ("cancelled", "deleted"):
                continue
            item = {
                "event_id": identity,
                "event_date": row["event_date"],
                "event_type": row.get("event_type", ""),
                "title": row.get("event_name", ""),
                "all_day": row.get("all_day") is True,
            }
            if item["all_day"]:
                if (
                    not row["event_date"]
                    <= target
                    < row.get(
                        "end_date", (parse_date(row["event_date"]) + timedelta(days=1)).isoformat()
                    )
                ):
                    continue
            else:
                start, end = instant(row.get("start_at")), instant(row.get("end_at"))
                if not start or not end:
                    historical_offset = row.get("timezone_offset")
                    if not historical_offset:
                        reasons.append("schedule_timezone_missing")
                        continue
                    try:
                        resolve_scope(row["event_date"], historical_offset)
                        start = instant(
                            f"{row['event_date']}T{row['start_time']}{historical_offset}"
                        )
                        end = instant(
                            f"{row.get('end_date', row['event_date'])}T"
                            f"{row['end_time']}{historical_offset}"
                        )
                    except (KeyError, TypeError, ValueError):
                        start = end = None
                if not start or not end or start >= end:
                    reasons.append("schedule_interval_ambiguous")
                    continue
                if start >= upper or end <= lower:
                    continue
                item.update(start_at_utc=timestamp(start), end_at_utc=timestamp(end))
            item["source_refs"] = collect_source_refs(row, "schedules", manifest)
            values.append(item)
        values.sort(key=lambda item: (item.get("start_at_utc", ""), item["event_id"]))
        return section(values, "partial", reasons)

    async def get_bedtime_reminder_inputs(
        self, uid: str, request_id: str, *, reminder_local_date: str, timezone_offset: str
    ) -> dict:
        day = parse_date(reminder_local_date)
        target = (day + timedelta(days=1)).isoformat()
        scope = resolve_scope((day + timedelta(days=2)).isoformat(), timezone_offset)
        read_at, manifest = utc_now(), {}
        schedules, debt = await asyncio.gather(
            self._optional(
                lambda: self._reminder_schedules(uid, target, timezone_offset, read_at, manifest),
            ),
            self._optional(
                lambda: self.build_sleep_debt(uid, reminder_local_date, read_at, manifest),
            ),
        )
        data = {
            "schedules": schedules,
            "sleep_debt": debt,
            "weather": section(status="unsupported", reasons=["source_not_configured"]),
        }
        result = self._envelope(uid, request_id, read_at, data, manifest)
        start, end = scope.bounds(1)
        result.update(
            reminder_local_date=reminder_local_date,
            target_local_date=target,
            timezone_offset=timezone_offset,
            target_window={"start_at_utc": timestamp(start), "end_at_utc": timestamp(end)},
        )
        return result

    async def _execution(self, uid: str, scope: Any, read_at: datetime, manifest: dict) -> dict:
        start_at, end_at = scope.bounds(28)
        sessions = await self.store.list_window_sessions(
            uid, scope.start(28).isoformat(), scope.end.isoformat(), start_at, end_at, read_at
        )
        events = await self.store.list_execution_events(uid, sessions, read_at)
        for source, rows in events.items():
            for row in rows:
                collect_source_refs(row, source, manifest)
        items, dates = [], {}
        for session in sessions:
            ended = instant(session.get("ended_at"))
            day = session.get("sleep_date_local")
            if not day and ended:
                day = attribute_sleep_date(ended, scope.offset)
            if not day or not scope.start(28).isoformat() <= day < scope.end.isoformat():
                continue
            dates[session["session_id"]] = day
            refs = collect_source_refs(session, "sessions", manifest)
            for item in summarize_execution(session, events["operations"], read_at):
                item["source_refs"] = refs
                items.append(item)
        value = {"sessions": items}
        for days in (7, 28):
            groups = {}
            for item in items:
                if dates[item["session_id"]] < scope.start(days).isoformat():
                    continue
                key = (item["phase"], item["resource_id"], item["resource_version"])
                groups.setdefault(key, []).append(item)
            totals = []
            for (phase, resource, version), group in groups.items():
                total = {
                    "start_local_date": scope.start(days).isoformat(),
                    "end_local_date_exclusive": scope.end.isoformat(),
                    "phase": phase,
                    "resource_id": resource,
                    "resource_version": version,
                    "exposed_session_count": len({row["session_id"] for row in group}),
                    "valid_night_count": len(
                        {dates[row["session_id"]] for row in group if "confirmed_on_sec" in row}
                    ),
                    "opportunity_policy_version": "unknown",
                    "source_refs": sorted({ref for row in group for ref in row["source_refs"]}),
                }
                if all("confirmed_on_sec" in row for row in group):
                    total["confirmed_on_sec_sum"] = sum(row["confirmed_on_sec"] for row in group)
                totals.append(total)
            value[f"totals_{days}d"] = totals
        return section(
            value,
            "partial",
            [
                "receipt_semantics_require_upstream_validation",
                "physical_measurements_and_opportunities_missing",
                "historical_session_ownership_required",
            ],
        )

    async def _plan(
        self, uid: str, plan_id: str, version: int, read_at: datetime, manifest: dict
    ) -> dict | None:
        row = await self.store.load_plan_revision(uid, plan_id, version, read_at)
        if row:
            if (
                row.get("uid") != uid
                or row.get("plan_id") != plan_id
                or row.get("version") != version
            ):
                raise DataError("PERMISSION_DENIED", "plan_version_owner_mismatch")
            collect_source_refs(row, "snapshots", manifest)
        return row

    @staticmethod
    def _ref(row: dict) -> dict:
        return _pick(row, "plan_id version snapshot_id content_hash date")

    async def _episode_diff(
        self,
        uid: str,
        session: dict,
        read_at: datetime,
        manifest: dict,
        generation: dict | None = None,
        revision_output: dict | None = None,
    ) -> dict:
        item = {
            "plan_id": session.get("plan_id", ""),
            "session_id": session["session_id"],
            "episode_id": session.get("episode_id", ""),
            "status": "partial",
            "diff": [],
            "execution_diff": [],
            "missing_reasons": [],
            "source_refs": collect_source_refs(session, "sessions", manifest),
        }
        reasons = item["missing_reasons"]
        if generation is None and session.get("generation_id"):
            try:
                _blueprint, generation = await self._blueprint(
                    uid, session["generation_id"], read_at, manifest
                )
            except DataError as exc:
                if exc.code != "NOT_FOUND":
                    raise
        outputs = [
            row
            for row in (generation or {}).get("output_plans", [])
            if row.get("plan_id") == item["plan_id"]
        ]
        if len(outputs) != 1:
            reasons.append("original_generation_reference_missing_or_ambiguous")
            return item
        original = await self._plan(uid, item["plan_id"], outputs[0]["version"], read_at, manifest)
        if not original or original.get("snapshot_id") != outputs[0].get("snapshot_id"):
            reasons.append("original_snapshot_missing_or_mismatched")
            return item
        item["original_plan_ref"] = self._ref(original)
        adopted = [
            row
            for row in session.get("adopted_versions", [])
            if row.get("applied") is True
            and instant(row.get("applied_at"))
            and visible_by(instant(row["applied_at"]), read_at)
        ]
        final_version = (
            max(adopted, key=lambda row: instant(row["applied_at"]))["version"]
            if adopted
            else session.get("plan_version")
        )
        if final_version is None:
            reasons.append("adopted_version_unknown")
            return item
        final = await self._plan(uid, item["plan_id"], final_version, read_at, manifest)
        if final is None:
            reasons.append("final_snapshot_missing")
            return item
        item["final_plan_ref"] = self._ref(final)
        revisions, seen, current = [], set(), final
        while current["version"] != original["version"]:
            if current["version"] in seen or len(seen) >= 100:
                reasons.append("revision_chain_cycle_or_budget")
                return item
            seen.add(current["version"])
            revisions.append(current)
            parent = current.get("base_plan_version")
            current = (
                await self._plan(uid, item["plan_id"], parent, read_at, manifest)
                if parent is not None
                else None
            )
            if current is None:
                reasons.append("revision_chain_missing")
                return item
        if revision_output is not None:
            for revision in revisions:
                converted = _pick(
                    revision, "version base_plan_version edit_source editor_id phases"
                )
                converted["lifecycle"] = revision.get("status", "unknown")
                created = instant(revision.get("created_at"))
                if created:
                    converted["created_at_utc"] = timestamp(created)
                converted["source_refs"] = collect_source_refs(revision, "snapshots", manifest)
                revision_output[revision["version"]] = converted
        events = await self.store.list_execution_events(uid, [session], read_at)
        operations = events["operations"]
        operations = [
            row
            for row in operations
            if row.get("session_id") == session["session_id"]
            and row.get("plan_id") == item["plan_id"]
        ]
        timed_operations = [
            row
            for row in operations
            if instant(row.get("executed_at")) and visible_by(instant(row["executed_at"]), read_at)
        ]
        if len(timed_operations) != len(operations):
            reasons.append("operation_time_missing_or_outside_read_scope")
        operations = timed_operations
        for source, rows in events.items():
            for row in rows:
                collect_source_refs(row, source, manifest)
        attempts = {
            "op:" + row["op_id"]
            for row in operations
            if row.get("origin") == "manual" and row.get("op_id")
        }
        attempts.update(
            "op:" + row["source_op_id"]
            if row.get("source_op_id")
            else "revision:" + str(row["version"])
            for row in revisions
            if row.get("edit_source") == "manual"
        )
        complete = session.get("evidence_complete") is True
        if attempts:
            item["user_adjusted"] = True
        elif complete:
            item["user_adjusted"] = False
        if complete:
            item["user_adjustment_count"] = len(attempts)
        else:
            reasons.append("manual_attempt_coverage_unknown")
        if not original.get("content_hash") or not final.get("content_hash"):
            reasons.append("snapshot_hash_missing")
        reasons.append("snapshot_hash_policy_unverified")
        try:
            item["diff"] = diff_plan_versions(original, final)
            adoption_time = max(
                (instant(row["applied_at"]) for row in adopted if row["version"] == final_version),
                default=None,
            )
            if adoption_time is None and session.get("plan_version") == final_version:
                adoption_time = instant(session.get("started_at"))
            if adoption_time:
                overlays = [
                    row
                    for row in operations
                    if instant(row["executed_at"]) >= adoption_time
                    and row.get("plan_version") == final_version
                ]
                item["execution_diff"] = compare_plan_execution(final, overlays, read_at)
                if any(row.get("plan_version") is None for row in operations):
                    reasons.append("operation_plan_version_unknown")
            else:
                reasons.append("final_version_adoption_time_missing")
        except ValueError:
            reasons.append("plan_shape_unsupported")
            return item
        path_origins = {}
        prior = original
        for revision in reversed(revisions):
            try:
                revision_changes = diff_plan_versions(prior, revision)
            except ValueError:
                reasons.append("intermediate_plan_shape_unsupported")
                return item
            for change in revision_changes:
                origin = revision.get("edit_source", "unknown")
                path_origins.setdefault(change["path"], set()).add(
                    origin if origin in ("manual", "automation", "safety") else "unknown"
                )
            prior = revision
        for change in item["diff"]:
            change["origins"] = sorted(path_origins.get(change["path"], {"unknown"}))
            change["source_refs"] = collect_source_refs(
                original, "snapshots", manifest
            ) + collect_source_refs(final, "snapshots", manifest)
        if operations and not all(row.get("received_at") for row in operations):
            reasons.append("read_visibility_best_effort")
        reasons.extend(["execution_segments_and_metadata_pending"])
        ended = instant(session.get("ended_at"))
        item["status"] = (
            "ongoing"
            if ended is None or not visible_by(ended, read_at)
            else "partial"
            if reasons
            else "complete"
        )
        return item

    async def build_plan_comparisons(
        self, uid: str, generation_id: str, read_at: datetime, manifest: dict
    ) -> dict:
        if not generation_id:
            return section(status="not_found", reasons=["no_review_generation"])
        _blueprint, generation = await self._blueprint(uid, generation_id, read_at, manifest)
        comparisons = []
        for output in generation.get("output_plans", []):
            initial = await self._plan(uid, output["plan_id"], output["version"], read_at, manifest)
            sessions = await self.store.list_sessions(uid, read_at, plan_id=output["plan_id"])
            item = {
                "generation_id": generation_id,
                "plan_id": output["plan_id"],
                "date": output.get("date", ""),
                "initial_plan_ref": self._ref(output),
                "status": "partial",
                "missing_reasons": ["comparison_metadata_and_segments_pending"],
            }
            if initial and initial.get("snapshot_id") == output.get("snapshot_id"):
                item["initial_phases"] = initial.get("phases", [])
            else:
                item["missing_reasons"].append("initial_snapshot_missing_or_mismatched")
            revisions = {}
            owned = [row for row in sessions if row.get("generation_id") == generation_id]
            item["episode_plan_diffs"] = [
                await self._episode_diff(uid, row, read_at, manifest, generation, revisions)
                for row in owned
            ]
            item["revisions"] = [revisions[version] for version in sorted(revisions)]
            item["sessions"] = []
            for row in owned:
                session = _pick(row, "session_id plan_id plan_version state")
                for source, target in (("started_at", "start_at_utc"), ("ended_at", "end_at_utc")):
                    when = instant(row.get(source))
                    if when and visible_by(when, read_at):
                        session[target] = timestamp(when)
                if row.get("episode_id"):
                    session["episode_ref"] = {
                        "episode_id": row["episode_id"],
                        "source": row.get("source", ""),
                    }
                item["sessions"].append(session)
            if any(row["status"] == "ongoing" for row in item["episode_plan_diffs"]):
                item["status"] = "ongoing"
            if not item["sessions"]:
                item["missing_reasons"].append("historically_owned_sessions_not_found")
            comparisons.append(item)
        return section(comparisons, "partial", ["comparison_metadata_and_segments_pending"])

    def _cursor_encode(self, payload: dict) -> str:
        secret = self.settings.generation_cursor_secret
        if len(secret.encode()) < 32:
            raise DataError("FAILED_PRECONDITION", "cursor_signing_key_not_configured")
        body = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode()
        signature = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        return body + "." + signature

    def _cursor_decode(self, token: str, binding: dict) -> dict:
        if len(token) > 8192:
            raise DataError("INVALID_ARGUMENT", "invalid_cursor")
        try:
            body, signature = token.rsplit(".", 1)
            payload = json.loads(base64.b64decode(body, altchars=b"-_", validate=True))
            expected = self._cursor_encode(payload).rsplit(".", 1)[1]
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            if any(payload.get(key) != value for key, value in binding.items()):
                raise ValueError
            started = instant(payload.get("read_at"))
            if (
                not started
                or not 0
                <= (utc_now() - started).total_seconds()
                <= self.settings.generation_cursor_ttl_sec
            ):
                raise ValueError
            if not isinstance(payload.get("last"), list) or len(payload["last"]) != 2:
                raise ValueError
            if not all(isinstance(value, str) for value in payload["last"]):
                raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DataError("INVALID_ARGUMENT", "invalid_cursor") from exc
        return payload

    async def get_episode_evidence(
        self,
        uid: str,
        episode_id: str,
        source: str,
        *,
        session_id: str = "",
        cursor: str = "",
        limit: int = 20,
    ) -> dict:
        if not episode_id or not source or not 1 <= limit <= 100:
            raise DataError("INVALID_ARGUMENT", "invalid_episode_or_limit")
        binding = {"uid": uid, "episode_id": episode_id, "source": source, "session_id": session_id}
        marker = self._cursor_decode(cursor, binding) if cursor else None
        read_at = instant(marker["read_at"]) if marker else utc_now()
        manifest = {}
        episode = await self.store.find_episode(uid, episode_id, source, read_at)
        if episode is None:
            raise DataError("NOT_FOUND", "episode_not_found")
        collect_source_refs(episode, "episodes", manifest)
        sessions = await self.store.list_sessions(
            uid, read_at, episode_id=episode_id, source=source, session_id=session_id
        )
        if session_id and not sessions:
            raise DataError("NOT_FOUND", "owned_session_not_found")
        sessions.sort(key=lambda row: (row["plan_id"], row["session_id"]))
        if marker:
            sessions = [
                row for row in sessions if [row["plan_id"], row["session_id"]] > marker["last"]
            ]
        page = sessions[:limit]
        values = [await self._episode_diff(uid, row, read_at, manifest) for row in page]
        reasons = [] if sessions else ["historically_owned_sessions_not_found"]
        result = {
            "plan_diffs": section(
                values,
                "partial"
                if reasons or any(row["status"] != "complete" for row in values)
                else "ok",
                reasons,
            ),
            "read_started_at_utc": timestamp(read_at),
            "source_manifest": list(manifest.values()),
        }
        if len(sessions) > limit:
            result["next_cursor"] = self._cursor_encode(
                {
                    **binding,
                    "read_at": timestamp(read_at),
                    "last": [page[-1]["plan_id"], page[-1]["session_id"]],
                }
            )
        return result
