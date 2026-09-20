"""Shared product/runtime contracts for the Ark S0 boundary.

The repository historically had several local shapes for projects, sessions and
retrieval rows.  This module is the single serialisable boundary between the
product layer and agentlab.  It deliberately contains data validation only;
storage, retrieval and UI concerns stay in their existing modules.

The models are additive and can be introduced at call boundaries without
changing the Markdown Vault or the existing session/index stores.
"""
from __future__ import annotations

import math
import re
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONTRACT_VERSION = "ark-contract-v1"


class RetrievalStrategy(str, Enum):
    """Product-facing retrieval strategy.

    ``shadow`` is intentionally a first-class value: it means the vector route
    is observed while lexical results remain the displayed candidates.
    """

    LEXICAL_ONLY = "lexical-only"
    VECTOR_ONLY = "vector-only"
    HYBRID = "hybrid"
    SHADOW = "shadow"


class RetrievalStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ProcessStatus(str, Enum):
    """Shared lifecycle status for ingest, indexing and enrichment stages."""

    ACCEPTED = "accepted"
    FETCHED = "fetched"
    PARSED = "parsed"
    CHUNKED = "chunked"
    INDEXED = "indexed"
    ENRICHED = "enriched"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"
    DEAD_LETTER = "dead-letter"


class CitationStatus(str, Enum):
    """Validity state for a citation used by an answer or artifact."""

    ACTIVE = "active"
    UNVERIFIED = "unverified"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ContractModel(BaseModel):
    """Base model: reject unknown fields at the boundary and serialise safely."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def _clean_id(value: str, field_name: str, *, max_length: int = 128) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(text) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters")
    return text


def _clean_optional_id(value: str | None, field_name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > 128:
        raise ValueError(f"{field_name} exceeds 128 characters")
    return text


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RetrievalScope(ContractModel):
    """Request scope shared by Vault, memory, session and RAG routes.

    An empty ``project_id``/``session_id`` means that the caller explicitly
    requested the global scope.  ``statuses=[]`` means "use the route's
    documented default"; it is different from silently carrying an old filter.
    """

    project_id: str = ""
    session_id: str = ""
    statuses: list[str] = Field(default_factory=list)
    include_archive: bool = False

    @field_validator("project_id", "session_id", mode="before")
    @classmethod
    def _ids(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)

    @field_validator("statuses", mode="before")
    @classmethod
    def _statuses(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("statuses must be a list of strings")
        out: list[str] = []
        for item in value:
            status = str(item).strip()
            if not status:
                continue
            if len(status) > 64:
                raise ValueError("status exceeds 64 characters")
            if status not in out:
                out.append(status)
        return out

    @classmethod
    def from_value(cls, value: "RetrievalScope | Mapping[str, Any] | None") -> "RetrievalScope":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("scope must be RetrievalScope, mapping, or None")
        return cls.model_validate(dict(value))


RequestScope = RetrievalScope


# A request-local scope is the only implicit context consumed by runtime tools.
# The default is created per read, so mutable lists cannot leak between requests.
_CURRENT_SCOPE: ContextVar[RetrievalScope | None] = ContextVar(
    "agentlab_retrieval_scope", default=None,
)


def current_retrieval_scope() -> RetrievalScope:
    """Return the bound request scope, or an explicit global scope."""

    value = _CURRENT_SCOPE.get()
    return value if value is not None else RetrievalScope()


def bind_retrieval_scope(scope: RetrievalScope | Mapping[str, Any] | None) -> Token:
    """Bind a validated scope for the current async/thread context."""

    return _CURRENT_SCOPE.set(RetrievalScope.from_value(scope))


def reset_retrieval_scope(token: Token) -> None:
    """Restore the previous request scope."""

    _CURRENT_SCOPE.reset(token)


class Provenance(ContractModel):
    """Source identity carried with every product-visible evidence object."""

    source: str
    ref: str
    source_id: str = ""
    project_id: str = ""
    session_id: str = ""
    captured_at: datetime | None = None
    version: str = ""

    @field_validator("source", "ref", mode="before")
    @classmethod
    def _required_text(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=4096)

    @field_validator("source_id", "project_id", "session_id", "version", mode="before")
    @classmethod
    def _optional_text(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)


class RetrievalItem(ContractModel):
    """Normalised item consumed by F2/F1/F3 and answer-level gates."""

    title: str = ""
    content: str = ""
    ref: str
    source: str = "vault"
    score: float = 0.0
    status: str = "active"
    project_id: str = ""
    session_id: str = ""
    provenance: list[Provenance] = Field(default_factory=list)
    context_of: list[str] = Field(default_factory=list)

    @field_validator("ref", mode="before")
    @classmethod
    def _ref(cls, value: Any) -> str:
        return _clean_id(value, "ref", max_length=4096)

    @field_validator("source", "status", mode="before")
    @classmethod
    def _source_status(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=128)

    @field_validator("project_id", "session_id", mode="before")
    @classmethod
    def _scope_ids(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("score must be numeric") from exc
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        return score

    @model_validator(mode="after")
    def _default_provenance(self) -> "RetrievalItem":
        if not self.provenance:
            self.provenance = [Provenance(
                source=self.source,
                ref=self.ref,
                project_id=self.project_id,
                session_id=self.session_id,
            )]
        return self


class RetrievalResult(ContractModel):
    """Stable retrieval envelope; no route may return an unlabelled result."""

    contract: str = CONTRACT_VERSION
    items: list[RetrievalItem] = Field(default_factory=list)
    strategy: RetrievalStrategy
    status: RetrievalStatus
    warnings: list[str] = Field(default_factory=list)
    scope: RetrievalScope = Field(default_factory=RetrievalScope)
    provenance: list[Provenance] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("warnings must be a list of strings")
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    @model_validator(mode="after")
    def _derive_provenance(self) -> "RetrievalResult":
        if not self.provenance:
            seen: set[tuple[str, str]] = set()
            for item in self.items:
                for prov in item.provenance:
                    key = (prov.source, prov.ref)
                    if key not in seen:
                        seen.add(key)
                        self.provenance.append(prov)
        if self.status == RetrievalStatus.AVAILABLE and self.warnings:
            # A route with warnings is not silently presented as pristine.
            self.status = RetrievalStatus.DEGRADED
        return self


RetrievalResponse = RetrievalResult
RouteResult = RetrievalResult


class Project(ContractModel):
    id: str
    title: str = ""
    name: str = ""
    mode: str = "general"
    goal: str = ""
    status: str = "active"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, value: Any) -> str:
        return _clean_id(value, "id", max_length=64)

    @field_validator("mode", "status", mode="before")
    @classmethod
    def _mode_status(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=64)

    @model_validator(mode="after")
    def _names(self) -> "Project":
        if not self.title and self.name:
            self.title = self.name
        if not self.name and self.title:
            self.name = self.title
        return self


class Source(ContractModel):
    ref: str
    title: str = ""
    kind: str
    scope: str = ""
    captured_at: datetime | None = None
    status: str = "active"
    provenance: list[Provenance] = Field(default_factory=list)

    @field_validator("ref", "kind", mode="before")
    @classmethod
    def _source_required(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=4096)

    @field_validator("status", mode="before")
    @classmethod
    def _source_status_field(cls, value: Any) -> str:
        return _clean_id(value, "status", max_length=64)

    @model_validator(mode="after")
    def _source_provenance(self) -> "Source":
        if not self.provenance:
            self.provenance = [Provenance(source=self.kind, ref=self.ref)]
        return self


class SourceDocument(ContractModel):
    """Normalised source identity shared by all ingestion adapters."""

    id: str
    source_kind: str
    source_ref: str
    title: str = ""
    content_hash: str = ""
    project_id: str = ""
    status: ProcessStatus = ProcessStatus.ACCEPTED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source_kind", "source_ref", mode="before")
    @classmethod
    def _required_source_text(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=4096)

    @field_validator("title", "content_hash", "project_id", mode="before")
    @classmethod
    def _optional_source_text(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)


class ProcessAttempt(ContractModel):
    """Retry-aware attempt envelope for one source processing run."""

    run_id: str
    attempt_id: str
    source_id: str
    status: ProcessStatus = ProcessStatus.ACCEPTED
    retry_count: int = 0
    error_code: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "attempt_id", "source_id", mode="before")
    @classmethod
    def _attempt_ids(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=256)

    @field_validator("retry_count", mode="before")
    @classmethod
    def _retry_count(cls, value: Any) -> int:
        try:
            number = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("retry_count must be an integer") from exc
        if number < 0:
            raise ValueError("retry_count must be non-negative")
        return number

    @field_validator("error_code", mode="before")
    @classmethod
    def _error_code(cls, value: Any) -> str:
        return _clean_optional_id(value, "error_code")


class StageResult(ContractModel):
    """Serializable result for one bounded processing stage."""

    stage_id: str
    stage_name: str
    status: ProcessStatus
    retryable: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    error_code: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_id", "stage_name", mode="before")
    @classmethod
    def _stage_text(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=256)

    @field_validator("warnings", "artifact_refs", mode="before")
    @classmethod
    def _string_list(cls, value: Any, info) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError(f"{info.field_name} must be a list of strings")
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    @field_validator("error_code", mode="before")
    @classmethod
    def _stage_error_code(cls, value: Any) -> str:
        return _clean_optional_id(value, "error_code")

    @model_validator(mode="after")
    def _validate_times(self) -> "StageResult":
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class Citation(ContractModel):
    """A source-bound reference that can be checked or revoked independently."""

    ref: str
    source_kind: str
    source_hash: str = ""
    project_id: str = ""
    session_id: str = ""
    status: CitationStatus = CitationStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ref", "source_kind", mode="before")
    @classmethod
    def _citation_required(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=4096)

    @field_validator("source_hash", "project_id", "session_id", mode="before")
    @classmethod
    def _citation_optional(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)


class Session(ContractModel):
    id: str
    project_id: str = ""
    goal: str = ""
    status: str = "active"
    events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("id", mode="before")
    @classmethod
    def _session_id(cls, value: Any) -> str:
        return _clean_id(value, "id", max_length=128)

    @field_validator("project_id", mode="before")
    @classmethod
    def _session_project(cls, value: Any) -> str:
        return _clean_optional_id(value, "project_id")

    @field_validator("status", mode="before")
    @classmethod
    def _session_status(cls, value: Any) -> str:
        return _clean_id(value, "status", max_length=64)


class Artifact(ContractModel):
    id: str
    project_id: str = ""
    kind: str
    path: str = ""
    source_refs: list[str] = Field(default_factory=list)
    status: str = "draft"
    created_by: str = "agent"
    session_id: str = ""
    provenance: list[Provenance] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("id", mode="before")
    @classmethod
    def _artifact_id(cls, value: Any) -> str:
        return _clean_id(value, "id", max_length=128)

    @field_validator("kind", "status", "created_by", mode="before")
    @classmethod
    def _artifact_text(cls, value: Any, info) -> str:
        return _clean_id(value, info.field_name, max_length=128)

    @field_validator("project_id", "session_id", mode="before")
    @classmethod
    def _artifact_scope(cls, value: Any, info) -> str:
        return _clean_optional_id(value, info.field_name)

    @field_validator("source_refs", mode="before")
    @classmethod
    def _refs(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("source_refs must be a list of strings")
        refs = []
        for ref in value:
            text = str(ref).strip()
            if text and text not in refs:
                refs.append(text)
        return refs


def strategy_from_modes(
    *,
    vector_enabled: bool = True,
    vector_mode: str = "shadow",
    lexical_mode: str = "on",
) -> RetrievalStrategy:
    """Map existing runtime switches to the frozen product strategy."""

    if not vector_enabled or str(vector_mode).lower() == "off":
        return RetrievalStrategy.LEXICAL_ONLY
    if str(vector_mode).lower() == "shadow":
        return RetrievalStrategy.SHADOW
    if str(lexical_mode).lower() == "off":
        return RetrievalStrategy.VECTOR_ONLY
    return RetrievalStrategy.HYBRID


def scoped_retrieval_result(
    items: Sequence[RetrievalItem | Mapping[str, Any]],
    *,
    strategy: RetrievalStrategy | str,
    status: RetrievalStatus | str = RetrievalStatus.AVAILABLE,
    warnings: Sequence[str] | None = None,
    scope: RetrievalScope | Mapping[str, Any] | None = None,
) -> RetrievalResult:
    """Validate/adapt arbitrary route rows into the frozen result envelope."""

    parsed: list[RetrievalItem] = []
    for item in items:
        parsed.append(item if isinstance(item, RetrievalItem)
                      else RetrievalItem.model_validate(dict(item)))
    return RetrievalResult(
        items=parsed,
        strategy=strategy,
        status=status,
        warnings=list(warnings or []),
        scope=RetrievalScope.from_value(scope),
    )


__all__ = [
    "CONTRACT_VERSION",
    "RetrievalStrategy",
    "RetrievalStatus",
    "RetrievalScope",
    "ProcessStatus",
    "CitationStatus",
    "RequestScope",
    "current_retrieval_scope",
    "bind_retrieval_scope",
    "reset_retrieval_scope",
    "Provenance",
    "RetrievalItem",
    "RetrievalResult",
    "RetrievalResponse",
    "RouteResult",
    "Project",
    "Source",
    "SourceDocument",
    "ProcessAttempt",
    "StageResult",
    "Citation",
    "Session",
    "Artifact",
    "strategy_from_modes",
    "scoped_retrieval_result",
]
