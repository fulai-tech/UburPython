"""Read-only change-stream invalidation, with fail-closed full-read fallback.

A process epoch intentionally invalidates persisted cursors after restart. No
source payloads, raw questionnaire text, or credentials are logged here.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class PreferenceRevision:
    """Drain Mongo's stream before reusing sources; deletions invalidate globally.

    Production sessions may have no uid, so ambiguous ownership invalidates all
    users. This costs a read but can never reuse another user's stale evidence.
    """

    def __init__(self, client, settings):
        self.client = client
        self.settings = settings
        self.stream = None
        self.epoch = uuid4().hex
        self.global_revision = 0
        self.users = {}
        self.changes = {}
        self.lock = asyncio.Lock()
        self.failed = client is None
        self.collections = [
            getattr(settings, f"somni_mongo_{name}_collection")
            for name in ("operations", "snapshots", "sessions")
        ]

    async def revision(self, uid: str) -> str | None:
        """Return a trusted cursor, or None when monitoring is unavailable."""
        if self.failed:
            return None
        async with self.lock:
            try:
                if self.stream is None:
                    self.stream = self.client[self.settings.somni_mongo_db].watch(
                        [{"$match": {"ns.coll": {"$in": self.collections}}}],
                        full_document="updateLookup",
                        max_await_time_ms=50,
                    )
                    await self.stream.__aenter__()
                # A bounded drain cannot certify freshness when the budget is hit.
                for _ in range(4096):
                    event = await self.stream.try_next()
                    if event is None:
                        if not self.stream.alive:
                            raise RuntimeError("preference_stream_closed")
                        return f"{self.epoch}:{self.global_revision}:{self.users.get(uid, 0)}"
                    if event.get("operationType") in {
                        "invalidate",
                        "drop",
                        "rename",
                        "dropDatabase",
                    }:
                        raise RuntimeError("preference_stream_invalidated")
                    owner = (event.get("fullDocument") or {}).get("uid")
                    if owner:
                        if len(self.users) >= 8192 and owner not in self.users:
                            self.global_revision += 1
                            self.users.clear()
                            self.changes.clear()
                        self.users[owner] = self.users.get(owner, 0) + 1
                        document = event.get("fullDocument") or {}
                        phase = document.get("phase") or (document.get("applied") or {}).get(
                            "phase"
                        )
                        phases = {phase} if phase in {"Unwind", "Soothe", "Guard", "Wake"} else None
                        history = self.changes.setdefault(owner, [])
                        history.append((self.users[owner], phases))
                        del history[:-256]
                    else:
                        self.global_revision += 1
                return None
            except (PyMongoError, RuntimeError, AttributeError, TypeError):
                self.failed = True
                logger.warning("Preference revision bypass: change stream unavailable")
                return None

    def changed_phases(self, uid: str, previous: str) -> set[str] | None:
        """Unknown ownership, epoch or truncated history requires all-phase refresh."""
        try:
            epoch, global_revision, user_revision, *_ = previous.split(":")
            cursor = int(user_revision)
            if epoch != self.epoch or int(global_revision) != self.global_revision:
                return None
            history = self.changes.get(uid, [])
            if not history or cursor < history[0][0] - 1:
                return None
            rows = [phases for seq, phases in history if seq > cursor]
            if not rows or any(phases is None for phases in rows):
                return None
            return set().union(*rows)
        except (ValueError, AttributeError):
            return None

    async def close(self) -> None:
        if self.stream is not None:
            await self.stream.close()
