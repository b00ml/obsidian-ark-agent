"""Bounded, serialisable processing-stage tracking for P0 pipelines."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from agentlab.contracts import ProcessStatus, StageResult


_RETRYABLE_CODES = {
    "AGENT_LLM_RATE", "AGENT_LLM_TIMEOUT", "AGENT_TOOL_TIMEOUT",
    "NETWORK", "TIMEOUT", "HTTP_5XX",
}


def _stamp() -> datetime:
    return datetime.now(timezone.utc)


class StageTracker:
    """Record one ordered set of stage attempts for a run."""

    def __init__(self, run_id: str, attempt_id: str, *, operation_id: str = "") -> None:
        self.run_id = str(run_id or "").strip()
        self.attempt_id = str(attempt_id or "").strip()
        self.operation_id = str(operation_id or "").strip()[:256]
        if not self.run_id or not self.attempt_id:
            raise ValueError("run_id and attempt_id are required")
        self._results: list[StageResult] = []

    @property
    def results(self) -> list[StageResult]:
        return list(self._results)

    def start(self, stage_id: str, stage_name: str,
              *, warnings: list[str] | None = None) -> StageResult:
        result = StageResult(
            stage_id=stage_id,
            stage_name=stage_name,
            status=ProcessStatus.ACCEPTED,
            warnings=warnings or [],
            metadata={
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                **({"operation_id": self.operation_id} if self.operation_id else {}),
            },
            started_at=_stamp(),
        )
        self._results.append(result)
        return result

    def record(self, stage_id: str, stage_name: str, *, status: ProcessStatus,
               warnings: list[str] | None = None,
               artifact_refs: list[str] | None = None,
               error_code: str = "", retryable: bool = False) -> StageResult:
        """Append one terminal stage without exposing a mutable intermediate.

        Adapter boundaries often have a single synchronous call for a stage.
        Recording that call through this helper preserves the same serialized
        contract as the context-manager path while keeping compatibility with
        existing handlers that return ordinary dictionaries.
        """
        result = self.start(stage_id, stage_name, warnings=warnings)
        result.error_code = str(error_code or "")[:256]
        result.retryable = bool(retryable)
        return self.finish(result, status=status, artifact_refs=artifact_refs)

    def finish(self, stage: StageResult, *, status: ProcessStatus,
               warnings: list[str] | None = None,
               artifact_refs: list[str] | None = None) -> StageResult:
        if stage not in self._results:
            raise ValueError("stage does not belong to tracker")
        stage.status = status
        stage.ended_at = _stamp()
        if warnings:
            stage.warnings = list(dict.fromkeys([*stage.warnings, *warnings]))
        if artifact_refs:
            stage.artifact_refs = list(dict.fromkeys([*stage.artifact_refs, *artifact_refs]))
        return stage

    def fail(self, stage: StageResult, error: Exception, *,
             retryable: bool | None = None, error_code: str = "") -> StageResult:
        code = error_code or str(getattr(error, "code", "STAGE_ERROR"))
        if retryable is None:
            retryable = code in _RETRYABLE_CODES
        stage.error_code = code
        stage.retryable = bool(retryable)
        return self.finish(stage, status=ProcessStatus.FAILED)

    def cancel(self, stage: StageResult, *, warning: str = "cancelled") -> StageResult:
        stage.retryable = False
        return self.finish(stage, status=ProcessStatus.CANCELLED,
                           warnings=[warning])

    def to_dict(self) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self._results]

    @contextmanager
    def stage(self, stage_id: str, stage_name: str) -> Iterator[StageResult]:
        """Track a stage and convert raised errors into a failed result."""
        result = self.start(stage_id, stage_name)
        try:
            yield result
        except asyncio.CancelledError:
            self.cancel(result)
            raise
        except Exception as exc:
            self.fail(result, exc)
            raise
        else:
            self.finish(result, status=ProcessStatus.COMPLETED)


__all__ = ["StageTracker"]
