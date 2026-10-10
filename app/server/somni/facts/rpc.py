from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError
from pymongo.errors import PyMongoError

from app.server.somni.facts.models import DataError
from app.server.somni.facts.service import GenerationDataService
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2 as pb
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2_grpc as rpc

Authorizer = Callable[[Any, str], Awaitable[bool]]


class GenerationDataRpc(rpc.GenerationDataServiceServicer):
    def __init__(
        self, service: GenerationDataService | None, authorizer: Authorizer | None = None
    ) -> None:
        self._service = service
        self._authorizer = authorizer

    async def _invoke(
        self,
        request: Any,
        context: Any,
        response_type: Any,
        call: Callable[[GenerationDataService, dict], Awaitable[dict]],
    ) -> Any:
        if self._authorizer is None:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "trusted_identity_provider_required"
            )
        try:
            uid = request.uid
            if not uid or len(uid) > 256 or uid != uid.strip():
                raise DataError("INVALID_ARGUMENT", "invalid_uid")
            if self._service is None:
                raise DataError("UNAVAILABLE", "generation_service_not_ready")
            async with asyncio.timeout(self._service.settings.generation_deadline_sec):
                if await self._authorizer(context, uid) is not True:
                    raise DataError("PERMISSION_DENIED", "uid_not_authorized")
                payload = MessageToDict(request, preserving_proto_field_name=True)
                if hasattr(request, "request_id") and not request.request_id.strip():
                    raise DataError("INVALID_ARGUMENT", "request_id_required")
                result = await call(self._service, payload)
                try:
                    response = ParseDict(result, response_type())
                except (ParseError, ValueError, TypeError) as exc:
                    raise DataError("INTERNAL", "response_contract_violation") from exc
                if response.ByteSize() > self._service.settings.generation_response_bytes:
                    raise DataError("RESOURCE_EXHAUSTED", "response_budget_exceeded")
                return response
        except DataError as exc:
            await context.abort(getattr(grpc.StatusCode, exc.code), exc.reason)
        except ValueError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_request")
        except TimeoutError:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "generation_deadline_exceeded")
        except PyMongoError:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "source_unavailable")
        except Exception:
            await context.abort(grpc.StatusCode.INTERNAL, "generation_request_failed")

    async def GetPlanInputs(self, request: Any, context: Any) -> pb.GetPlanInputsRes:
        async def call(service: GenerationDataService, payload: dict) -> dict:
            return await service.get_plan_inputs(**payload)

        return await self._invoke(request, context, pb.GetPlanInputsRes, call)

    async def GetProfileInputs(self, request: Any, context: Any) -> pb.GetProfileInputsRes:
        async def call(service: GenerationDataService, payload: dict) -> dict:
            if payload.get("purpose") not in (
                "initial",
                "weekly",
                "correction",
                "plan_generation",
                "profile_refresh",
            ):
                raise DataError("INVALID_ARGUMENT", "invalid_purpose")
            if not payload.get("review_end_local_date") or not payload.get("timezone_offset"):
                raise DataError("INVALID_ARGUMENT", "review_scope_required")
            return await service.get_profile_inputs(**payload)

        return await self._invoke(request, context, pb.GetProfileInputsRes, call)

    async def GetEpisodeEvidence(self, request: Any, context: Any) -> pb.GetEpisodeEvidenceRes:
        async def call(service: GenerationDataService, payload: dict) -> dict:
            if not payload.get("episode_id") or not payload.get("source"):
                raise DataError("INVALID_ARGUMENT", "episode_and_source_required")
            return await service.get_episode_evidence(**payload)

        return await self._invoke(request, context, pb.GetEpisodeEvidenceRes, call)

    async def GetBedtimeReminderInputs(
        self, request: Any, context: Any
    ) -> pb.GetBedtimeReminderInputsRes:
        async def call(service: GenerationDataService, payload: dict) -> dict:
            if not payload.get("reminder_local_date") or not payload.get("timezone_offset"):
                raise DataError("INVALID_ARGUMENT", "reminder_scope_required")
            return await service.get_bedtime_reminder_inputs(**payload)

        return await self._invoke(request, context, pb.GetBedtimeReminderInputsRes, call)
