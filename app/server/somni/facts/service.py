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
    normalize_daily_metrics,
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
from app.server.somni.facts.sleep_debt import calculate_sleep_debt
from app.server.somni.facts.store import FactsStore
from app.server.somni.facts.time import attribute_sleep_date, instant, parse_date, resolve_scope


def _pick(row: dict, fields: str) -> dict:
    return {key: row[key] for key in fields.split() if key in row and row[key] is not None}


def collect_source_refs(row: dict, source: str, manifest: dict) -> list[str]:
    identifier = (
        row.get("_id")
        or row.get("snapshot_id")
        or row.get("generation_id")
        or row.get("observation_id")
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


class GenerationDataService:
    def __init__(self, client: Any, settings: Settings, store: FactsStore | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else FactsStore(client, settings)

    async def _permissions(self, uid: str) -> dict:
        row = await self.store.load_permissions(uid)
        if not row or row.get("status") not in (1, "active"):
            raise DataError("PERMISSION_DENIED", "account_not_authorized")
        return row.get("privacy_switches") if isinstance(row.get("privacy_switches"), dict) else {}

    @staticmethod
    def _allowed(permissions: dict, source: str) -> bool:
        return permissions.get(source) is True

    async def _optional(self, allowed: bool, query: Callable[[], Awaitable[dict]]) -> dict:
        if not allowed:
            return section(status="unsupported", reasons=["consent_not_granted"])
        try:
            return await query()
        except DataError as exc:
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
        self, uid: str, snapshot_id: str, read_at: datetime, manifest: dict, end_date: str = ""
    ) -> dict:
        short, long = await self.store.load_profile(uid, snapshot_id, read_at, end_date)
        if short is None:
            if snapshot_id:
                raise DataError("NOT_FOUND", "profile_not_found")
            return section(status="not_found", reasons=["no_prior_profile"])
        collect_source_refs(short, "profile_short", manifest)
        if long is None:
            return section(status="partial", reasons=["hash_linked_long_profile_missing"])
        collect_source_refs(long, "profile_long", manifest)
        profile = _pick(short, " ".join(PROFILE_FIELDS))
        profile.update(
            profile_snapshot_id=short.get("snapshot_id", ""),
            long_term_profile=long.get("long_term_profile"),
        )
        for source_key, target in (
            ("data_cutoff_at", "data_cutoff_at_utc"),
            ("generated_at", "generated_at_utc"),
        ):
            value = instant(short.get(source_key))
            if value:
                profile[target] = timestamp(value)
        if long.get("uid") != uid or long.get("content_hash") != short.get(
            "long_term_content_hash"
        ):
            raise DataError("PERMISSION_DENIED", "profile_link_mismatch")
        profile["content_hash"] = canonical_hash(profile)
        try:
            profile = validate_profile(profile, uid)
        except DataError as exc:
            if exc.code == "PERMISSION_DENIED":
                raise
            return section(status="partial", reasons=[exc.reason])
        return section(profile)

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
    ) -> dict:
        has_profile_input = bool(profile_snapshot_id) or inline_profile is not None
        if bool(profile_snapshot_id) == (inline_profile is not None) and has_profile_input:
            raise DataError("INVALID_ARGUMENT", "exactly_one_profile_input_required")
        if feedback_record_id and not target_local_date:
            raise DataError("INVALID_ARGUMENT", "feedback_target_date_required")
        if not has_profile_input and not target_local_date:
            raise DataError("INVALID_ARGUMENT", "profile_input_or_target_date_required")
        if feedback_record_id != feedback_record_id.strip():
            raise DataError("INVALID_ARGUMENT", "invalid_feedback_record_id")
        read_at, manifest = utc_now(), {}
        permissions = await self._permissions(uid)
        if not has_profile_input:
            profile = section(status="not_found", reasons=["profile_input_not_requested"])
        elif self._allowed(permissions, "profiles"):
            profile = (
                section(validate_profile(inline_profile, uid))
                if inline_profile is not None
                else await self._profile(uid, profile_snapshot_id, read_at, manifest)
            )
        else:
            profile = section(status="unsupported", reasons=["consent_not_granted"])
        if self._allowed(permissions, "plans"):
            blueprint, _row = await self._blueprint(uid, previous_generation_id, read_at, manifest)
        else:
            blueprint = section(status="unsupported", reasons=["consent_not_granted"])
        data = {"profile": profile, "previous_blueprint": blueprint}
        if target_local_date:
            data["target_plan"] = await self._target_plan(
                uid,
                target_local_date,
                read_at,
                manifest,
                plans_allowed=self._allowed(permissions, "plans"),
            )
        if feedback_record_id:
            data["feedback"] = await self._feedback_observation(
                uid,
                feedback_record_id,
                target_local_date,
                read_at,
                manifest,
                observations_allowed=self._allowed(permissions, "observations"),
            )
        return self._envelope(uid, request_id, read_at, data, manifest)

    async def _feedback_observation(
        self,
        uid: str,
        observation_id: str,
        target_local_date: str,
        read_at: datetime,
        manifest: dict,
        *,
        observations_allowed: bool,
    ) -> dict:
        if not observations_allowed:
            return section(status="unsupported", reasons=["consent_not_granted"])
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
        plans_allowed: bool,
    ) -> dict:
        if not plans_allowed:
            return section(status="unsupported", reasons=["consent_not_granted"])
        try:
            target_date = parse_date(target_local_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise DataError("INVALID_ARGUMENT", "invalid_target_local_date") from exc
        # Resolve the latest effective plan through the unique active pointer visible at read_at.
        rows = await self.store.load_active_plan(uid, target_date, read_at)
        if not rows:
            return section(status="not_found", reasons=["active_plan_not_found"])
        if len(rows) != 1:
            raise DataError("FAILED_PRECONDITION", "active_plan_ambiguous")
        pointer = rows[0]
        if pointer.get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "active_plan_owner_mismatch")
        plan_id = str(pointer.get("plan_id") or "")
        weekly_generation_id = str(pointer.get("weekly_generation_id") or "")
        version = pointer.get("version")
        if not plan_id or not weekly_generation_id or not isinstance(version, int) or version < 1:
            raise DataError("FAILED_PRECONDITION", "active_plan_link_incomplete")
        blueprint_section, generation = await self._blueprint(
            uid, weekly_generation_id, read_at, manifest
        )
        if generation is None or blueprint_section["status"] not in ("ok", "partial"):
            raise DataError("FAILED_PRECONDITION", "active_plan_generation_unavailable")
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
        lifecycle = revision.get("lifecycle")
        if isinstance(lifecycle, dict):
            lifecycle = lifecycle.get("status")
        if lifecycle is None:
            lifecycle = revision.get("status")
        if lifecycle not in (None, "", "active"):
            raise DataError("FAILED_PRECONDITION", "active_plan_snapshot_not_active")
        if revision.get("plan_start_local_date") not in (None, "", target_date):
            raise DataError("PERMISSION_DENIED", "active_plan_date_mismatch")
        refs = collect_source_refs(pointer, "plans", manifest)
        refs.extend(collect_source_refs(revision, "snapshots", manifest))
        value = _pick(
            revision,
            "plan_id version snapshot_id content_hash base_plan_version edit_source phases",
        )
        value["lifecycle"] = str(lifecycle or "active")
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

    async def _questionnaires(self, uid: str, read_at: datetime, manifest: dict) -> dict:
        rows = await self.store.load_questionnaires(uid, read_at)
        values, reasons = [], []
        for row in rows:
            item = _pick(row, "answer_id survey_code survey_version")
            if not all(item.get(key) for key in ("answer_id", "survey_code", "survey_version")):
                reasons.append("questionnaire_version_missing")
                continue
            for source, target in (
                ("completed_at", "completed_at_utc"),
                ("received_at", "received_at_utc"),
            ):
                when = instant(row.get(source))
                if when:
                    item[target] = timestamp(when)
            item["answers"] = [
                _pick(
                    answer, "question_id question_version field_code value unit scale option_codes"
                )
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
        rows = await self.store.list_sleep_episodes(
            uid, scope.start(28).isoformat(), scope.end.isoformat(), read_at
        )
        activity = await self.store.load_daily_metrics(
            uid, scope.start(28).isoformat(), scope.end.isoformat(), read_at
        )
        episodes, reasons = (
            [],
            [
                "raw_signal_and_event_aggregation_pending",
                "read_visibility_best_effort",
                "partial_metric_mapping",
            ],
        )
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
                or end > read_at
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
        metrics = normalize_daily_metrics(rows) + normalize_daily_metrics(activity, activity=True)
        value = {"episodes": episodes, "series": build_series(metrics, scope)}
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
        data_cutoff_at_utc: str = "",
    ) -> dict:
        if purpose not in ("initial", "weekly", "correction", "plan_generation"):
            raise DataError("INVALID_ARGUMENT", "invalid_purpose")
        if purpose == "weekly" and not review_generation_id:
            raise DataError("INVALID_ARGUMENT", "weekly_generation_required")
        scope = resolve_scope(review_end_local_date, timezone_offset)
        read_at, manifest = utc_now(), {}
        if data_cutoff_at_utc:
            cutoff = instant(data_cutoff_at_utc)
            if cutoff is None or cutoff > read_at:
                raise DataError("INVALID_ARGUMENT", "invalid_data_cutoff")
        else:
            cutoff = read_at
        allowed_sections = {
            "questionnaires",
            "prior_profile",
            "facts",
            "observations",
            "plan_comparisons",
            "execution",
            "sleep_debt",
        }
        if requested_sections and not set(requested_sections) <= allowed_sections:
            raise DataError("INVALID_ARGUMENT", "unknown_requested_section")
        permissions = await self._permissions(uid)
        permissions["execution"] = self._allowed(permissions, "health") and self._allowed(
            permissions, "plans"
        )
        tasks = {
            "sleep_debt": (
                "health",
                lambda: self.build_sleep_debt(
                    uid, (scope.end - timedelta(days=1)).isoformat(), cutoff, manifest
                ),
            ),
            "questionnaires": (
                "questionnaires",
                lambda: self._questionnaires(uid, read_at, manifest),
            ),
            "prior_profile": (
                "profiles",
                lambda: self._profile(
                    uid, previous_profile_snapshot_id, read_at, manifest, review_end_local_date
                ),
            ),
            "facts": ("health", lambda: self.build_facts(uid, scope, read_at, manifest)),
            "observations": (
                "observations",
                lambda: self._observations(uid, scope, read_at, manifest),
            ),
            "plan_comparisons": (
                "execution",
                lambda: self.build_plan_comparisons(uid, review_generation_id, read_at, manifest),
            ),
            "execution": ("execution", lambda: self._execution(uid, scope, read_at, manifest)),
        }
        if requested_sections:
            tasks = {key: value for key, value in tasks.items() if key in requested_sections}
        if data_cutoff_at_utc and set(tasks) != {"sleep_debt"}:
            raise DataError("INVALID_ARGUMENT", "historical_cutoff_requires_sleep_debt_only")
        results = await asyncio.gather(
            *(
                self._optional(self._allowed(permissions, permission), query)
                for permission, query in tasks.values()
            )
        )
        data = dict(zip(tasks, results, strict=True))
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
        return calculate_sleep_debt(rows, record_date, cutoff)

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
        permissions = await self._permissions(uid)
        schedules, debt = await asyncio.gather(
            self._optional(
                self._allowed(permissions, "calendar"),
                lambda: self._reminder_schedules(uid, target, timezone_offset, read_at, manifest),
            ),
            self._optional(
                self._allowed(permissions, "health"),
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
            and instant(row["applied_at"]) <= read_at
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
            if instant(row.get("executed_at")) and instant(row["executed_at"]) <= read_at
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
            "ongoing" if ended is None or ended > read_at else "partial" if reasons else "complete"
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
                    if when and when <= read_at:
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
        permissions = await self._permissions(uid)
        if not self._allowed(permissions, "health") or not self._allowed(permissions, "plans"):
            return {
                "plan_diffs": section(status="unsupported", reasons=["consent_not_granted"]),
                "read_started_at_utc": timestamp(utc_now()),
            }
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
