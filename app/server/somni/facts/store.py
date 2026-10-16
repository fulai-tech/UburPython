from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.codec_options import CodecOptions

from app.core.config import Settings
from app.server.somni.facts.models import DataError
from app.server.somni.facts.sleep_debt import cohort, select_daily
from app.server.somni.facts.time import current_generation_read, instant, observation_reference


class FactsStore:
    def __init__(self, client: Any, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def _read(
        self,
        name: str,
        query: dict,
        fields: str,
        sort: list[tuple[str, int]] | None = None,
        limit: int | None = None,
        *,
        missing_reasons: list[str] | None = None,
    ) -> list[dict]:
        if self._client is None:
            raise DataError("UNAVAILABLE", "mongo_not_configured")
        collection = self._client[self._settings.somni_mongo_db][
            getattr(self._settings, f"somni_mongo_{name}_collection")
        ].with_options(codec_options=CodecOptions(tz_aware=True, tzinfo=UTC))
        maximum = limit or self._settings.generation_query_limit
        if current_generation_read():
            query = self._current_query(query)
        cutoff = self._visibility_cutoff(query)
        projection = {field: 1 for field in fields.split()}
        if cutoff is not None or current_generation_read():
            projection.update(
                received_at=1,
                create_time=1,
                update_time=1,
                created_at=1,
                generated_at=1,
                completed_at=1,
            )
        cursor = collection.find(query, projection).max_time_ms(
            self._settings.generation_query_timeout_ms
        )
        if sort:
            cursor = cursor.sort(sort)
        rows = await cursor.limit(maximum + 1).to_list(length=maximum + 1)
        if len(rows) > maximum and limit is None:
            raise DataError("RESOURCE_EXHAUSTED", "source_query_budget_exceeded")
        if current_generation_read():
            rows = [self._normalize_times(row) for row in rows]
        if cutoff is not None:
            rows = [self._normalize_times(row) for row in rows]
            verified = []
            for row in rows:
                try:
                    self._validate_time(row, cutoff)
                except DataError as exc:
                    if missing_reasons is None or exc.reason not in (
                        "source_time_unverified",
                        "historical_version_unavailable",
                    ):
                        raise
                    reason = f"{name}_{exc.reason}"
                    if reason not in missing_reasons:
                        missing_reasons.append(reason)
                    continue
                required_time = {
                    "user_profiles_short_terms": "generated_at",
                    "generations": "generated_at",
                    "snapshots": "created_at",
                }.get(name)
                if required_time and instant(row.get(required_time)) is None:
                    raise DataError("FAILED_PRECONDITION", "source_time_unverified")
                verified.append(row)
            rows = verified
            if not rows and query.get("status") in ("active", "committed"):
                historical_query = {key: value for key, value in query.items() if key != "status"}
                candidates = await (
                    collection.find(historical_query, projection)
                    .max_time_ms(self._settings.generation_query_timeout_ms)
                    .limit(1)
                    .to_list(length=1)
                )
                if candidates:
                    self._validate_time(self._normalize_times(candidates[0]), cutoff)
                    raise DataError("FAILED_PRECONDITION", "historical_version_unavailable")
        return rows[:maximum]

    @staticmethod
    def _current_query(query: dict) -> dict:
        result = {}
        for key, value in query.items():
            if key in {"received_at", "created_at", "generated_at", "completed_at", "update_time"}:
                continue
            if key in {"$and", "$or"}:
                children = [FactsStore._current_query(child) for child in value]
                if key == "$or" and any(not child for child in children):
                    continue
                children = [child for child in children if child]
                if children:
                    result[key] = children
            else:
                result[key] = value
        return result

    @staticmethod
    def _normalize_times(row: dict) -> dict:
        row = dict(row)
        if row.get("created_at") is None and row.get("create_time") is not None:
            row["created_at"] = row["create_time"]
        if row.get("received_at") is None:
            row["received_at"] = row.get("update_time") or row.get("create_time")
        return row

    @staticmethod
    def _visibility_cutoff(query: dict) -> datetime | None:
        received = query.get("received_at")
        if isinstance(received, dict) and isinstance(received.get("$lte"), datetime):
            return received["$lte"]
        for operator in ("$and", "$or"):
            for child in query.get(operator, []):
                cutoff = FactsStore._visibility_cutoff(child)
                if cutoff is not None:
                    return cutoff
        return None

    @staticmethod
    def _validate_time(row: dict, cutoff: datetime) -> None:
        if current_generation_read():
            return
        received = instant(row.get("received_at"))
        if received is None:
            raise DataError("FAILED_PRECONDITION", "source_time_unverified")
        if received > cutoff:
            raise DataError("FAILED_PRECONDITION", "historical_version_unavailable")
        for field in ("update_time", "created_at", "generated_at", "completed_at"):
            if row.get(field) is None:
                continue
            observed = instant(row[field])
            if observed is None:
                raise DataError("FAILED_PRECONDITION", "source_time_unverified")
            if observed > cutoff:
                raise DataError("FAILED_PRECONDITION", "historical_version_unavailable")

    @staticmethod
    def _visible(query: dict, read_at: datetime) -> dict:
        if current_generation_read():
            return FactsStore._current_query(query)
        return {
            **query,
            "$and": [
                *query.get("$and", []),
                {
                    "$or": [
                        {"received_at": {"$lte": read_at}},
                        {"received_at": {"$not": {"$type": "date"}}},
                    ]
                },
                *[
                    {"$or": [{field: {"$lte": read_at}}, {field: {"$not": {"$type": "date"}}}]}
                    for field in ("created_at", "generated_at", "completed_at")
                ],
            ],
        }

    async def load_sleep_debt_rows(
        self, uid: str, record_date: str, cutoff: datetime
    ) -> list[dict]:
        query = {
            "user_id": uid,
            "status": 1,
            "record_date": record_date,
            **({} if current_generation_read() else {"received_at": {"$lte": cutoff}}),
        }
        current = await self._read("wearable_daily", query, self._daily_fields())
        chosen = select_daily(current, None if current_generation_read() else cutoff).get(
            record_date
        )
        if chosen is None:
            return current
        query.update(record_date={"$lt": record_date}, source=chosen["source"])
        collection = self._client[self._settings.somni_mongo_db][
            self._settings.somni_mongo_wearable_daily_collection
        ].with_options(codec_options=CodecOptions(tz_aware=True, tzinfo=UTC))
        cursor = (
            collection.find(query, {key: 1 for key in self._daily_fields().split()})
            .sort([("record_date", -1), ("revision", -1), ("_id", -1)])
            .max_time_ms(self._settings.generation_query_timeout_ms)
            .batch_size(128)
        )
        rows, group, day, valid_count, scanned = list(current), [], None, 0, len(current)
        try:
            async for row in cursor:
                self._validate_time(row, cutoff)
                scanned += 1
                if scanned > self._settings.generation_query_limit:
                    raise DataError("RESOURCE_EXHAUSTED", "sleep_history_budget_exceeded")
                if day is not None and row.get("record_date") != day:
                    selected = select_daily(
                        group, None if current_generation_read() else cutoff
                    ).get(day)
                    if selected and cohort(selected) == cohort(chosen):
                        rows.append(selected)
                        valid_count += 1
                    if valid_count == 14:
                        return rows
                    group = []
                day = row.get("record_date")
                group.append(row)
            if group:
                selected = select_daily(group, None if current_generation_read() else cutoff).get(
                    day
                )
                if selected and cohort(selected) == cohort(chosen):
                    rows.append(selected)
            return rows
        finally:
            await cursor.close()

    async def load_reminder_schedules(
        self, uid: str, dates: list[str], read_at: datetime, lower: datetime, upper: datetime
    ) -> list[dict]:
        return await self._read(
            "schedules",
            self._visible(
                {
                    "uid": uid,
                    "$and": [
                        {
                            "$or": [
                                {"event_date": {"$in": dates}},
                                {"start_at": {"$lt": upper}, "end_at": {"$gt": lower}},
                                {"event_date": {"$lte": dates[-1]}, "end_date": {"$gt": dates[-1]}},
                            ]
                        }
                    ],
                },
                read_at,
            ),
            "event_id event_date event_type event_name start_time end_time start_at end_at "
            "end_date all_day status revision timezone_offset received_at update_time",
            [("event_date", 1), ("start_time", 1), ("_id", 1)],
        )

    async def list_window_sessions(
        self,
        uid: str,
        start_date: str,
        end_date: str,
        start_at: datetime,
        end_at: datetime,
        read_at: datetime,
    ) -> list[dict]:
        return await self._read(
            "sessions",
            self._visible(
                {
                    "uid": uid,
                    **({} if current_generation_read() else {"started_at": {"$lte": read_at}}),
                    "$and": [
                        {
                            "$or": [
                                {"sleep_date_local": {"$gte": start_date, "$lt": end_date}},
                                {
                                    "sleep_date_local": {"$exists": False},
                                    "ended_at": {"$gte": start_at, "$lt": end_at},
                                },
                            ]
                        }
                    ],
                },
                read_at,
            ),
            "uid session_id episode_id source plan_id plan_version started_at ended_at "
            "sleep_date_local state received_at evidence_complete",
        )

    async def load_questionnaires(
        self, uid: str, read_at: datetime, answer_id: str = ""
    ) -> list[dict]:
        if answer_id and (answer_id != answer_id.strip() or len(answer_id) > 256):
            raise DataError("INVALID_ARGUMENT", "invalid_answer_id")
        query = {"uid": uid, "is_completed": True}
        if answer_id:
            identifiers = [{"answer_id": answer_id}]
            if ObjectId.is_valid(answer_id):
                identifiers.append({"_id": ObjectId(answer_id)})
            query["$or"] = identifiers
        answers = await self._read(
            "answers",
            self._visible(query, read_at),
            "answer_id survey_code survey_version answer_revision completed_at received_at answers",
            [("completed_at", -1), ("answer_revision", -1), ("_id", -1)],
        )
        selected = {}
        answers.sort(
            key=lambda row: (
                instant(row.get("completed_at"))
                or instant(row.get("update_time"))
                or instant(row.get("create_time"))
                or datetime.min.replace(tzinfo=UTC),
                str(row.get("_id", "")),
            ),
            reverse=True,
        )
        for answer in answers:
            if not answer.get("answer_id") and answer.get("_id") is not None:
                answer["answer_id"] = str(answer["_id"])
            if not answer.get("completed_at"):
                answer["completed_at"] = answer.get("update_time") or answer.get("create_time")
            code = answer.get("survey_code")
            if code and code not in selected:
                selected[code] = answer
        for answer in selected.values():
            if answer.get("answer_id") and answer.get("answer_revision") is not None:
                answer["results"] = await self._read(
                    "quiz_results",
                    self._visible(
                        {
                            "uid": uid,
                            "answer_id": answer["answer_id"],
                            "answer_revision": answer["answer_revision"],
                        },
                        read_at,
                    ),
                    "instrument_id rule_version value unit received_at",
                )
        return list(selected.values())

    async def load_quiz_questions(self, question_ids: list[str]) -> dict[str, dict]:
        ids = list(dict.fromkeys(str(qid) for qid in question_ids if qid))
        if not ids:
            return {}
        if self._client is None:
            raise DataError("UNAVAILABLE", "mongo_not_configured")
        oids = [ObjectId(qid) for qid in ids if ObjectId.is_valid(qid)]
        clauses: list[dict] = []
        if oids:
            clauses.append({"_id": {"$in": oids}})
        clauses.append({"_id": {"$in": ids}})
        query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
        collection = self._client[self._settings.somni_mongo_db][
            self._settings.somni_mongo_questions_collection
        ]
        cursor = collection.find(query, {"input_type": 1, "options": 1}).max_time_ms(
            self._settings.generation_query_timeout_ms
        )
        docs = await cursor.to_list(length=len(ids) + len(oids))
        return {str(doc["_id"]): doc for doc in docs if doc.get("_id") is not None}

    async def load_profile(
        self, uid: str, snapshot_id: str, read_at: datetime, end_date: str = ""
    ) -> tuple[dict | None, dict | None]:
        query = {"uid": uid}
        if snapshot_id:
            query["$and"] = [
                {"$or": [{"profile_snapshot_id": snapshot_id}, {"snapshot_id": snapshot_id}]}
            ]
        else:
            query.update(
                {
                    "record_date": {"$lte": end_date},
                    "status": "active",
                }
            )
        rows = await self._read(
            "user_profiles_short_terms",
            self._visible(query, read_at),
            "uid profile_snapshot_id snapshot_id snapshot_revision schema_version "
            "content_hash long_term_content_hash "
            "record_date generated_at status short_term_profile rolling_baseline "
            "profile_quality based_on received_at",
            [("record_date", -1), ("snapshot_revision", -1), ("_id", -1)],
            limit=2,
        )
        if not rows:
            return None, None
        if len(rows) > 1 and snapshot_id:
            raise DataError("FAILED_PRECONDITION", "profile_snapshot_ambiguous")
        short = rows[0]
        if short.get("profile_snapshot_id") and short.get("snapshot_id") not in (
            None,
            "",
            short["profile_snapshot_id"],
        ):
            raise DataError("FAILED_PRECONDITION", "profile_snapshot_alias_mismatch")
        short["snapshot_id"] = short.get("profile_snapshot_id") or short.get("snapshot_id", "")
        if not short.get("long_term_content_hash") or not short.get("schema_version"):
            return short, None
        long_rows = await self._read(
            "user_profile_long_terms",
            self._visible(
                {
                    "uid": uid,
                    "schema_version": short["schema_version"],
                    "content_hash": short["long_term_content_hash"],
                },
                read_at,
            ),
            "uid schema_version content_hash status long_term_profile received_at",
            [("_id", -1)],
            limit=1,
        )
        return short, long_rows[0] if long_rows else None

    async def load_blueprint(self, uid: str, generation_id: str, read_at: datetime) -> dict | None:
        rows = await self._read(
            "generations",
            self._visible(
                {"uid": uid, "generation_id": generation_id, "status": "committed"}, read_at
            ),
            "uid generation_id status generated_at received_at raw_plan decision_record "
            "profile_ref output_plans content_hash",
            limit=1,
        )
        return rows[0] if rows else None

    async def load_active_plan(self, uid: str, target_date: str, read_at: datetime) -> list[dict]:
        """Load canonical active pointers; independent callers retain their semantics."""
        return await self._read(
            "plans",
            self._visible({"uid": uid, "date": target_date, "status": "active"}, read_at),
            "uid date plan_id version weekly_generation_id generation_id received_at",
            [("version", -1), ("_id", -1)],
            limit=2,
        )

    async def load_stored_current_plan(
        self, uid: str, target_date: str, read_at: datetime
    ) -> list[dict]:
        """Load unique current plans, accepting the production no-status shape."""
        return await self._read(
            "plans",
            self._visible(
                {
                    "uid": uid,
                    "date": target_date,
                    "status": {"$exists": False},
                },
                read_at,
            ),
            "uid date plan_id version weekly_generation_id generation_id received_at "
            "phases device_id scope name status",
            [("version", -1), ("_id", -1)],
            limit=2,
        )

    async def load_latest_active_plan_date_before(
        self, uid: str, target_date: str, read_at: datetime
    ) -> str | None:
        """Find the newest active target date strictly before a weekly cycle."""
        rows = await self._read(
            "plans",
            self._visible({"uid": uid, "date": {"$lt": target_date}, "status": "active"}, read_at),
            "uid date",
            [("date", -1), ("_id", -1)],
            limit=1,
        )
        value = rows[0].get("date") if rows else None
        return value if isinstance(value, str) and value < target_date else None

    async def has_plan_history_before(self, uid: str, target_date: str, read_at: datetime) -> bool:
        """Detect prior effective or disabled plans before allowing first-generation fallback."""
        rows = await self._read(
            "plans",
            self._visible(
                {
                    "uid": uid,
                    "date": {"$lt": target_date},
                    "status": {"$in": ["active", "disabled"]},
                },
                read_at,
            ),
            "uid date status",
            [("date", -1), ("_id", -1)],
            limit=1,
        )
        return bool(rows)

    async def list_sleep_episodes(
        self,
        uid: str,
        start: str,
        end: str,
        read_at: datetime,
        *,
        missing_reasons: list[str] | None = None,
    ) -> list[dict]:
        dates = {"$gte": start, "$lt": end}
        wearable = await self._read(
            "wearable_daily",
            self._visible({"user_id": uid, "status": 1, "record_date": dates}, read_at),
            self._daily_fields(),
            missing_reasons=missing_reasons,
        )
        radar = await self._read(
            "records",
            self._visible({"uid": uid, "record_date": dates}, read_at),
            "episode_id source revision record_date timezone_offset main_sleep "
            "selection_policy_version raw_data quality received_at metric_windows",
            missing_reasons=missing_reasons,
        )
        return [{**row, "source_type": "wearable_daily"} for row in wearable] + [
            {**row, "source_type": "records"} for row in radar
        ]

    @staticmethod
    def _daily_fields() -> str:
        return (
            "episode_id source revision record_date activity_date timezone_offset "
            "sleep physiology activity quality data_gaps device_algorithm definition_version "
            "hrv_method main_sleep selection_policy_version source_upload_record_ids received_at "
            "metric_windows"
        )

    async def load_daily_metrics(
        self,
        uid: str,
        start: str,
        end: str,
        read_at: datetime,
        *,
        missing_reasons: list[str] | None = None,
    ) -> list[dict]:
        return await self._read(
            "wearable_daily",
            self._visible(
                {"user_id": uid, "status": 1, "activity_date": {"$gte": start, "$lt": end}}, read_at
            ),
            self._daily_fields(),
            missing_reasons=missing_reasons,
        )

    async def list_daily_sleep_reports(
        self,
        uid: str,
        start: str,
        end: str,
        read_at: datetime,
        *,
        missing_reasons: list[str] | None = None,
    ) -> list[dict]:
        return await self._read(
            "sleep_reports",
            self._visible(
                {"uid": uid, "record_date": {"$gte": start, "$lt": end}},
                read_at,
            ),
            "uid reports_id record_date language overnight_vitals",
            [("record_date", 1), ("language", 1), ("reports_id", 1), ("_id", 1)],
            missing_reasons=missing_reasons,
        )

    async def load_upload_status(self, uid: str, ids: list[str], read_at: datetime) -> list[dict]:
        if not ids:
            return []
        return await self._read(
            "uploads",
            self._visible({"user_id": uid, "record_id": {"$in": ids}}, read_at),
            "record_id source data_type status error_code received_at processed_at",
        )

    async def list_observations(
        self, uid: str, start: str, end: str, read_at: datetime
    ) -> list[dict]:
        return await self._read(
            "observations",
            self._visible(
                {
                    "uid": uid,
                    "status": "active",
                    "local_date": {"$gte": start, "$lt": end},
                    "$and": [
                        {
                            "$or": [
                                {"valid_from": {"$lte": observation_reference(read_at)}},
                                {"valid_from": {"$exists": False}},
                            ]
                        },
                        {
                            "$or": [
                                {"valid_to": {"$gt": observation_reference(read_at)}},
                                {"valid_to": None},
                            ]
                        },
                    ],
                },
                read_at,
            ),
            "observation_id kind field_code value unit scale local_date target_dates episode_ref "
            "plan_refs occurred_at received_at valid_from valid_to supersedes_id quality",
        )

    async def load_feedback_observation(
        self, uid: str, observation_id: str, read_at: datetime
    ) -> dict | None:
        rows = await self._read(
            "observations",
            self._visible(
                {
                    "uid": uid,
                    "observation_id": observation_id,
                    "status": "active",
                    "$and": [
                        {
                            "$or": [
                                {"valid_from": {"$lte": observation_reference(read_at)}},
                                {"valid_from": {"$exists": False}},
                            ]
                        },
                        {
                            "$or": [
                                {"valid_to": {"$gt": observation_reference(read_at)}},
                                {"valid_to": None},
                            ]
                        },
                    ],
                },
                read_at,
            ),
            "uid observation_id kind field_code value unit scale local_date record_date "
            "target_dates episode_ref plan_refs occurred_at received_at valid_from valid_to "
            "quality",
            limit=1,
        )
        return rows[0] if rows else None

    async def load_plan_revision(
        self, uid: str, plan_id: str, version: int, read_at: datetime
    ) -> dict | None:
        rows = await self._read(
            "snapshots",
            self._visible({"uid": uid, "plan_id": plan_id, "version": version}, read_at),
            "uid plan_id version snapshot_id content_hash generation_id base_plan_version "
            "edit_source editor_id created_at received_at phases status lifecycle source_op_id "
            "plan_start_local_date timezone_offset execution_time date create_time device_id "
            "apply_status trigger_reason request_id",
            limit=2,
        )
        if len(rows) > 1:
            raise DataError("FAILED_PRECONDITION", "plan_revision_ambiguous")
        return rows[0] if rows else None

    async def find_episode(
        self, uid: str, episode_id: str, source: str, read_at: datetime
    ) -> dict | None:
        for name, owner in (("wearable_daily", "user_id"), ("records", "uid")):
            query = {owner: uid, "episode_id": episode_id, "source": source}
            if name == "wearable_daily":
                query["status"] = 1
            rows = await self._read(
                name,
                self._visible(query, read_at),
                "episode_id source record_date session_ids received_at",
                limit=1,
            )
            if rows:
                return rows[0]
        return None

    async def load_preference_sources(
        self, uid: str, end_local_date: str, timezone_offset: str
    ) -> dict[str, list[dict]]:
        """Read a bounded learning window; reception times never restrict visibility.

        Operations without an owned session remain unverified in aggregation. A
        historical device identifier alone is not sufficient authorization.
        """
        from datetime import timedelta

        from app.server.somni.facts.time import parse_date, parse_offset

        end_date = parse_date(end_local_date)
        zone = parse_offset(timezone_offset)
        first = end_date - timedelta(days=14)
        lower = datetime.combine(first, datetime.min.time(), zone).astimezone(UTC)
        # Include the morning completing the last business night.
        upper = datetime.combine(end_date, datetime.min.time(), zone).astimezone(UTC)
        upper += timedelta(hours=18)
        operations = await self._read(
            "operations",
            {"uid": uid, "executed_at": {"$gte": lower, "$lt": upper}},
            "uid op_id request_id device_id sn session_id plan_id plan_version phase "
            "node_key resource_id origin method payload applied success executed_at",
            [("executed_at", 1), ("_id", 1)],
        )
        snapshots = await self._read(
            "snapshots",
            {
                "uid": uid,
                "$or": [
                    {
                        "plan_start_local_date": {
                            "$gte": first.isoformat(),
                            "$lt": end_date.isoformat(),
                        }
                    },
                    {"date": {"$gte": first.isoformat(), "$lt": end_date.isoformat()}},
                ],
            },
            "uid plan_id version snapshot_id content_hash base_plan_version "
            "edit_source editor_id created_at create_time phases status lifecycle apply_status "
            "plan_start_local_date date timezone_offset source_op_id device_id",
            [("plan_id", 1), ("version", 1), ("_id", 1)],
        )
        # Sessions have no persisted uid in production. Authorization comes from
        # the owned immutable revision, never from a device identifier alone.
        plan_ids = sorted(
            {row["plan_id"] for row in snapshots if row.get("uid") == uid and row.get("plan_id")}
        )
        sessions = []
        if plan_ids:
            # A duplicated plan id owned by someone else is not an authorization.
            foreign = await self._read(
                "snapshots",
                {"plan_id": {"$in": plan_ids}, "uid": {"$ne": uid}},
                "plan_id uid",
                limit=1,
            )
            if foreign:
                raise DataError("PERMISSION_DENIED", "preference_plan_owner_ambiguous")
            sessions = await self._read(
                "sessions",
                {
                    "plan_id": {"$in": plan_ids},
                    "started_at": {"$gte": lower, "$lt": upper},
                    "$or": [{"uid": uid}, {"uid": {"$exists": False}}],
                },
                "uid session_id device_id sn plan_id plan_version started_at ended_at "
                "state evidence_complete adopted_versions executed_nodes timezone "
                "end_reason previewing total_nodes node_index phase phase_started_at",
                [("started_at", 1), ("_id", 1)],
            )
        return {"operations": operations, "snapshots": snapshots, "sessions": sessions}

    async def list_sessions(
        self,
        uid: str,
        read_at: datetime,
        *,
        episode_id: str = "",
        source: str = "",
        session_id: str = "",
        plan_id: str = "",
    ) -> list[dict]:
        query = {"uid": uid}
        if episode_id:
            query.update({"episode_id": episode_id, "source": source})
        if session_id:
            query["session_id"] = session_id
        if plan_id:
            query["plan_id"] = plan_id
        return await self._read(
            "sessions",
            self._visible(query, read_at),
            "uid session_id episode_id source generation_id plan_id plan_version started_at "
            "ended_at state received_at evidence_complete adopted_versions device_id phase "
            "node_index next_unexecuted_node_key",
            [("plan_id", 1), ("session_id", 1)],
        )

    async def list_execution_events(
        self, uid: str, sessions: list[dict], read_at: datetime
    ) -> dict[str, list[dict]]:
        ids = [
            row["session_id"] for row in sessions if row.get("uid") == uid and row.get("session_id")
        ]
        if not ids:
            return {name: [] for name in ("operations", "downlinks", "events", "session_payloads")}
        query = self._visible({"uid": uid, "session_id": {"$in": ids}}, read_at)
        projections = {
            "operations": "op_id session_id plan_id plan_version origin method payload applied "
            "success executed_at received_at ack reason phase node_key node_index "
            "resource_id resource_version parameters",
            "downlinks": "downlink_id session_id payload sent_at received_at ack status",
            "events": "event_id session_id type event_timestamp received_at duration_sec "
            "detail related_event_id origin phase node_index resource_id",
            "session_payloads": "session_id plan_id plan_version phases received_at applied_at",
        }
        return {name: await self._read(name, query, fields) for name, fields in projections.items()}

    async def iter_signal_windows(
        self, uid: str, session: dict, start: datetime, end: datetime, read_at: datetime
    ) -> list[dict]:
        if session.get("uid") != uid or not session.get("session_id"):
            raise DataError("PERMISSION_DENIED", "historical_session_owner_unverified")
        return await self._read(
            "telemetry",
            self._visible(
                {
                    "uid": uid,
                    "session_id": session["session_id"],
                    "start_at": {"$lt": end},
                    "end_at": {"$gt": start},
                },
                read_at,
            ),
            "session_id start_at end_at ts window_ms sample_count data unit method "
            "definition_version calibration weighting received_at",
        )

    async def load_execution_metadata(self, uid: str, snapshot: dict) -> dict:
        if snapshot.get("uid") != uid:
            raise DataError("PERMISSION_DENIED", "snapshot_owner_mismatch")
        return {"devices": [], "resources": [], "missing_reasons": ["frozen_metadata_missing"]}
