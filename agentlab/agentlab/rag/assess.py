"""充分性评估 / LLM 自 rerank（docs/04 §4.3，F9）。

`RAGAssessor.assess(query, items)`：判断检索结果是否足以回答问题。
- 足够 → sufficient=True，agent 直接综合回答（引用 ref）；
- 不足且可改写 → action="reformulate"，返回 reformulated_query 供再次召回；
- 严重不足 → action="insufficient"，如实"信息不足"，避免编造。

结构化输出（JSON）强制二次校验（AGENTS.md），失败抛 AGENT_GUARDRAIL。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from agentlab.core.errors import AgentError
from agentlab.core.guardrails import ensure_structured
from agentlab.core.llm import LLMProvider
from agentlab.core.message import Message
from agentlab.prompts import load_prompt
from agentlab.rag.recall import RecallItem
from agentlab.contracts import RetrievalScope, current_retrieval_scope


@dataclass
class Sufficiency:
    sufficient: bool
    action: str              # "answer" | "reformulate" | "insufficient"
    reformulated_query: str = ""
    message: str = ""
    refs: list[str] | None = None
    answerability: str = "answerable"

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient, "action": self.action,
            "reformulated_query": self.reformulated_query,
            "message": self.message, "refs": self.refs or [],
            "answerability": self.answerability,
        }


@dataclass(frozen=True)
class AnswerGateResult:
    """Deterministic pre/post generation safety decision.

    The LLM may judge semantic sufficiency, but it cannot override source
    boundaries, forbidden refs, inactive candidates or an explicitly absent
    answer.  This result is intentionally serialisable for trace/eval output.
    """

    decision: str
    allowed_refs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    abstained: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision == "answerable"

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "allowed_refs": list(self.allowed_refs),
                "reasons": list(self.reasons), "abstained": self.abstained}


_ABSTENTION_PREFIX_RE = re.compile(
    r"^\s*(?:#+[^\n]*\n\s*)?(?:(?:当前|目前|现有|本轮|就目前)\s*)?"
    r"(?:"
    r"(?:候选|相关|现有)[^\n]{0,18}(?:未发现|没有发现|没有支持|未覆盖)|"
    r"(?:资料|信息|证据|检索结果)(?:[^\n]{0,18})?(?:不足|不够|缺失|为空|没有|未覆盖)|"
    r"(?:候选|相关|现有|资料|信息|证据|检索结果)[^\n]{0,18}不支持"
    r"(?:回答|作答|确认|判断|确定|给出结论|该问题)|"
    r"(?:无法|不能|不可|不应)(?:[^\n]{0,18})?(?:确认|回答|作答|判断|确定|提供)|"
    r"(?:没有找到|未找到|找不到)(?:[^\n]{0,18})(?:资料|证据|来源)|"
    r"需要(?:更多|补充)(?:[^\n]{0,18})(?:资料|信息|来源|证据)"
    r")",
    re.IGNORECASE,
)
_BOUNDED_LIMITATION_RE = re.compile(
    r"(?:资料|信息|证据|检索结果)(?:[^\n]{0,24})?(?:不足|不够|缺失|未覆盖|有限)|"
    r"(?:本轮|当前|目前)(?:[^\n]{0,24})(?:未覆盖|无法确认|不能确认|不作判断)|"
    r"(?:只|仅)(?:能|做|可|会)?[^\n]{0,18}(?:复述|转述|说明|回答|概括|总结|列出)",
    re.IGNORECASE,
)
_WIKILINK_RE = re.compile(
    r"\[\[([^\]|\n]+?)(?:\|[^\]\n]*)?\]\]"
)
_WIKILINK_RENDER_RE = re.compile(
    r"\[\[([^\]|\n]+?)(?:\|([^\]\n]*))?\]\]"
)
_MD_REF_RE = re.compile(
    r"(?<![\w/])([^\s<>\[\](){}，。；;：:、\"'“”‘’]+\.md(?:#[^\s<>\[](){}，。；;：:、\"'“”‘’]+)?)",
    re.IGNORECASE,
)
_CONTEXTUAL_MD_REF_RE = re.compile(
    r"(?:依据|来源|引用|参考|参见|见|详见|来自|出处|证据)"
    r"(?:[ \t]*(?:资料|文档|笔记|链接|文件))?"
    # A delimiter is required.  Without it, ordinary prose such as
    # “来源版（文件夹总结…）” starts at “来源” and swallows text up to the
    # next ``.md`` as a fictional citation.
    r"[ \t]*(?:是|为)?[ \t]*[:：][ \t]*"
    r"(?P<ref>[^\n。；;，,!?！？<>()[\]{}]*?\.md"
    r"(?:#[^\n。；;，,!?！？<>()[\]{} ]+)?)",
    re.IGNORECASE,
)
_FENCED_CODE_RE = re.compile(
    r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$"
)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_TOOL_CALL_RE = re.compile(
    r"\b(?:rag_[a-z_]+|vault_[a-z_]+|memory_[a-z_]+|bili_[a-z_]+|"
    r"article_[a-z_]+|context_[a-z_]+)\s*\([^()\n]*\)",
    re.IGNORECASE,
)
_CITATION_CONTEXT_RE = re.compile(
    r"(?:依据|来源|引用|参考|参见|见|详见|来自|出处|证据)"
    r"(?:[ \t]*(?:资料|文档|笔记|链接|文件))?"
    r"[ \t]*(?:是|为)?[ \t]*[:：]?[ \t]*(?:[-*][ \t]*)?$",
    re.IGNORECASE,
)


def _normalise_ref(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("<>`[]")


def _item_block(items: list[RecallItem]) -> str:
    if not items:
        return "（无检索结果）"
    lines = []
    for i, it in enumerate(items, 1):
        # ``source`` is route metadata; ``ref`` is the sole citation identity.
        # Keep them separate so the assessor cannot copy a synthetic
        # ``source/ref`` value that the evidence gate must reject.
        ref = json.dumps(str(it.ref or ""), ensure_ascii=False)
        source = json.dumps(str(it.source or ""), ensure_ascii=False)
        lines.append(
            f"[{i}] ref={ref}; source={source}; title={it.title}\n"
            f"    {it.content[:300]}"
        )
    return "\n".join(lines)


def _schema_ok(obj: Any) -> bool:
    """结构化输出谓词，并校验 action/sufficient 的跨字段契约。"""
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("sufficient"), bool):
        return False
    action = obj.get("action")
    if action not in ("answer", "reformulate", "insufficient"):
        return False
    if action == "answer" and obj["sufficient"] is not True:
        return False
    if action != "answer" and obj["sufficient"] is not False:
        return False
    if action == "reformulate" and not str(obj.get("reformulated_query") or "").strip():
        return False
    refs = obj.get("refs", [])
    if not isinstance(refs, list) or not all(isinstance(ref, str) and ref.strip() for ref in refs):
        return False
    answerability = obj.get("answerability", "answerable")
    return answerability in {"answerable", "absent", "conflicting", "scope_denied"}


def _ref_allowed(ref: str, candidates: set[str]) -> bool:
    """Allow an entry-level citation to point at one of its concrete chunks."""
    value = str(ref or "").strip()
    return bool(value) and any(_ref_matches(value, candidate) for candidate in candidates)


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _ref_matches(expected: str, actual: str) -> bool:
    expected = _normalise_ref(expected)
    actual = _normalise_ref(actual)
    if not expected or not actual:
        return False
    if (expected == actual or actual.startswith(expected + ":ch") or
            actual.startswith(expected + "#") or
            expected.startswith(actual + ":ch") or
            expected.startswith(actual + "#")):
        return True

    # A wikilink normally contains the note title (``[[Redis缓存]]``), while
    # retrieval refs carry a vault directory and ``.md`` suffix.  Compare the
    # entry basename as a compatibility identity, but keep fragments/chunk
    # ids out of the comparison.  Full paths still take precedence above.
    def identities(value: str) -> tuple[str, set[str]]:
        entry = value.split("#", 1)[0].split(":ch", 1)[0].rstrip("/")
        basename = entry.rsplit("/", 1)[-1]
        return entry, {
            entry,
            basename,
            basename[:-3] if basename.lower().endswith(".md") else basename,
        }

    expected_entry, expected_ids = identities(expected)
    actual_entry, actual_ids = identities(actual)
    # Never equate two distinct full paths merely because their filenames
    # match.  Basename/stem compatibility is only for a bare wikilink title.
    expected_bare = "/" not in expected_entry
    actual_bare = "/" not in actual_entry
    return (expected_bare or actual_bare) and bool(expected_ids & actual_ids)


def _is_placeholder_citation(value: str) -> bool:
    """Ignore citation-shaped template prose, not a real source identity."""
    token = _normalise_ref(value).strip().lower()
    if token in {"wikilink", "link", "source", "ref", "reference", "来源", "引用"}:
        return True
    return bool(re.search(r"\{[^{}]+\}|\.\.\.|…", token))


def _looks_like_abstention(answer: str) -> bool:
    """Detect a full refusal without flagging qualified uncertainty in an answer.

    Models often say a proposition is unsupported while giving a useful,
    cited conclusion, for example "资料并不支持把两者当作互斥选项".
    That is not a refusal to answer. Treat only a refusal-led first sentence
    about insufficient evidence or inability to answer as full abstention;
    long answers remain checked by citation and assessment gates.
    """
    text = str(answer or "").strip()
    if not text:
        return False
    if _ABSTENTION_PREFIX_RE.search(text):
        return True
    return False


def _has_bounded_limitation(answer: str) -> bool:
    """Return whether the answer explicitly limits unsupported claims."""
    return bool(_BOUNDED_LIMITATION_RE.search(str(answer or "")))


def _mask_matches(text: str, pattern: re.Pattern[str]) -> str:
    """Mask non-prose spans while preserving offsets for context checks."""
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _plain_ref(value: str) -> str:
    """Normalise a Markdown path captured outside an explicit wikilink."""
    return _normalise_ref(value).strip(
        "`*_~.,，。；;：:、!?！？\"'“”‘’"
    )


def _answer_citation_tokens(
    answer: str, candidate_refs: Iterable[str] = ()
) -> list[str]:
    """Extract explicit citations and conservatively classify plain Markdown paths.

    Wikilinks are explicit even when they do not end in ``.md``.  Plain paths
    are accepted only when they match retrieved candidates or follow an
    unambiguous citation cue.  Code/examples and tool calls are masked first;
    this prevents paths in ``rag_retrieve(...)`` or template snippets from
    becoming out-of-scope citations.
    """
    text = str(answer or "")
    visible = _mask_matches(text, _FENCED_CODE_RE)
    visible = _mask_matches(visible, _TOOL_CALL_RE)
    candidates = [
        _plain_ref(ref) for ref in candidate_refs
        if _plain_ref(ref)
    ]

    # Wikilinks are explicit citations by contract, including when a user
    # formats the link with inline backticks.  Fenced blocks were masked first
    # so example links in documentation still remain non-evidence.
    tokens = [
        value for value in (_normalise_ref(match) for match in _WIKILINK_RE.findall(visible))
        if value and not _is_placeholder_citation(value)
    ]

    # A real source path is often rendered as inline code (`` `wiki/a.md` ``).
    # Count it only when it exactly matches a retrieved candidate; arbitrary
    # code/template paths remain masked and cannot create an out-of-scope hit.
    for match in _INLINE_CODE_RE.finditer(visible):
        value = _plain_ref(match.group(0).strip("`"))
        if not value or ".md" not in value.lower():
            continue
        if any(_ref_matches(value, candidate) or _ref_matches(candidate, value)
               for candidate in candidates):
            tokens.append(value)

    visible = _mask_matches(visible, _INLINE_CODE_RE)
    # A wikilink may contain spaces and Chinese punctuation.  Mask its whole
    # span before ordinary-path extraction so the fallback scanner cannot leak
    # a suffix such as ``空格.md`` as a second citation.
    prose = _mask_matches(visible, _WIKILINK_RE)
    # Preserve a full path with spaces when it follows a clear citation cue.
    # The matched span is masked before the generic scanner to avoid emitting
    # both the full path and a suffix token.
    contextual_spans: list[tuple[int, int]] = []
    for match in _CONTEXTUAL_MD_REF_RE.finditer(prose):
        token = _plain_ref(match.group("ref"))
        if token:
            tokens.append(token)
            contextual_spans.append((match.start("ref"), match.end("ref")))
    if contextual_spans:
        chars = list(prose)
        for start, end in contextual_spans:
            chars[start:end] = " " * (end - start)
        prose = "".join(chars)

    candidate_set = set(candidates)
    for match in _MD_REF_RE.finditer(prose):
        token = _plain_ref(match.group(1))
        if not token or _is_placeholder_citation(token):
            continue
        candidate_match = any(
            _ref_matches(token, candidate) or _ref_matches(candidate, token)
            for candidate in candidate_set
        )
        prefix = prose[max(0, match.start() - 96):match.start()]
        contextual = bool(_CITATION_CONTEXT_RE.search(prefix))
        if candidate_match or contextual:
            tokens.append(token)

    # Generic extraction cannot retain spaces, so explicitly look for a
    # candidate's entry path as a final compatibility path (e.g. a plain
    # ``依据 wiki/带 空格.md`` citation against a chunk candidate).
    for candidate in candidate_set:
        entry = candidate.split("#", 1)[0]
        if ".md" not in entry.lower():
            continue
        if re.search(re.escape(entry), prose):
            tokens.append(entry)
    return list(dict.fromkeys(token for token in tokens if token))


def sanitize_generated_answer(answer: str, candidate_refs: Iterable[str]) -> str:
    """Remove citation semantics from wikilinks outside this retrieval set.

    Candidate content often contains ordinary Obsidian links to related notes.
    They are useful prose, but emitting them unchanged makes the answer gate
    treat them as citations.  Keep links whose target resolves to a candidate;
    render every other link as inline code so the visible answer remains useful
    without claiming an out-of-scope source.
    """
    candidates = [_plain_ref(ref) for ref in candidate_refs if _plain_ref(ref)]

    def replace(match: re.Match[str]) -> str:
        target = _normalise_ref(match.group(1))
        if target and any(_ref_matches(target, ref) or _ref_matches(ref, target)
                          for ref in candidates):
            return match.group(0)
        label = str(match.group(2) or target).strip()
        return f"`{label}`" if label else ""

    return _WIKILINK_RENDER_RE.sub(replace, str(answer or ""))


def evaluate_generated_answer(
    answer: str,
    candidate_refs: Iterable[str],
    *,
    answerability: str = "answerable",
    assessment: str = "",
    require_citation: bool = True,
    allow_bounded_partial: bool = False,
) -> AnswerGateResult:
    """Run the deterministic post-generation answer gate.

    This is deliberately conservative: it checks citation boundaries and the
    explicit abstention contract, but does not pretend that string matching can
    prove semantic claim coverage.  Semantic sufficiency remains the job of
    ``rag_assess``/human review.
    """
    refs = [_normalise_ref(ref) for ref in candidate_refs if _normalise_ref(ref)]
    refs = list(dict.fromkeys(refs))
    answerability = str(answerability or "answerable").strip().lower()
    assessment = str(assessment or "").strip().lower()
    abstained = _looks_like_abstention(answer)
    if answerability in {"absent", "unanswerable", "unknown", "insufficient", "scope_denied"}:
        reason = (
            "scope_denied" if answerability == "scope_denied" else
            "answerability_insufficient" if answerability == "insufficient" else
            "answerability_absent"
        )
        return AnswerGateResult("insufficient", reasons=(reason,), abstained=abstained)
    if answerability == "conflicting":
        return AnswerGateResult("conflicting", reasons=("conflicting_evidence",), abstained=abstained)
    assessment_requires_abstention = assessment in {"insufficient", "reformulate"}
    if assessment_requires_abstention and not abstained and not allow_bounded_partial:
        return AnswerGateResult("insufficient", reasons=("assessment_requires_abstention",))
    if not refs:
        return AnswerGateResult("insufficient", reasons=("no_usable_evidence",), abstained=abstained)

    tokens = _answer_citation_tokens(answer, refs)
    cited = [ref for ref in refs if any(_ref_matches(ref, token) or _ref_matches(token, ref)
                                        for token in tokens)]
    outside = [token for token in tokens if not any(
        _ref_matches(token, ref) or _ref_matches(ref, token) for ref in refs
    )]
    if outside:
        return AnswerGateResult("insufficient", tuple(cited), ("citation_out_of_scope",), abstained)
    if abstained:
        return AnswerGateResult("insufficient", tuple(cited), ("explicit_abstention",), True)
    if require_citation and not cited:
        return AnswerGateResult("insufficient", (), ("citation_missing",))
    if assessment_requires_abstention:
        # ``answerability`` is an explicit semantic safety label.  A partial
        # answer may relax an assessor's coverage verdict, but must never
        # override an absent/insufficient/conflicting/scope-denied verdict.
        if (allow_bounded_partial and answerability == "answerable"
                and cited and _has_bounded_limitation(answer)):
            return AnswerGateResult("answerable", tuple(cited), ("bounded_partial",))
        return AnswerGateResult("insufficient", tuple(cited), ("assessment_requires_abstention",))
    return AnswerGateResult("answerable", tuple(cited))


@dataclass
class AnswerEvidence:
    """Per-run evidence ledger used by Runner's post-generation gate."""

    require_citation: bool = True
    candidate_refs: set[str] = None  # type: ignore[assignment]
    retrieval_calls: int = 0
    answerability: str = "answerable"
    assessment: str = ""
    allow_bounded_partial: bool = False

    def __post_init__(self) -> None:
        if self.candidate_refs is None:
            self.candidate_refs = set()

    def observe(self, tool_name: str, content: str) -> None:
        if tool_name not in {"rag_retrieve", "rag_assess"}:
            return
        try:
            value = json.loads(content or "")
        except (TypeError, ValueError):
            return
        if tool_name == "rag_retrieve":
            self.retrieval_calls += 1
            self.assessment = ""  # a fresh retrieval supersedes an old reformulation verdict
            # A new candidate set also supersedes a semantic verdict attached
            # to the previous set.  Keeping ``insufficient`` here made a
            # successful follow-up retrieval remain fail-closed forever.
            self.answerability = "answerable"
            rows = value.get("items", []) if isinstance(value, dict) else value
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status") or "active").strip().lower()
                    if status in {"active", "current", "available"}:
                        ref = _normalise_ref(row.get("ref"))
                        if ref:
                            self.candidate_refs.add(ref)
            if isinstance(value, dict):
                status = str(value.get("status") or "").strip().lower()
                if status == "scope_denied":
                    self.answerability = "scope_denied"
        else:
            action = str(value.get("action") or "").strip().lower() if isinstance(value, dict) else ""
            self.assessment = action
            explicit = str(value.get("answerability") or "").strip().lower() if isinstance(value, dict) else ""
            if explicit in {
                "absent", "unanswerable", "unknown", "insufficient",
                "conflicting", "scope_denied",
            }:
                self.answerability = explicit

    def check(self, answer: str) -> AnswerGateResult | None:
        if not self.retrieval_calls:
            return None
        return evaluate_generated_answer(
            answer, self.candidate_refs, answerability=self.answerability,
            assessment=self.assessment, require_citation=self.require_citation,
            allow_bounded_partial=self.allow_bounded_partial,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_calls": self.retrieval_calls,
            "candidate_refs": sorted(self.candidate_refs),
            "answerability": self.answerability,
            "assessment": self.assessment,
            "allow_bounded_partial": self.allow_bounded_partial,
        }


def evaluate_answer_gate(
    query: str,
    items: list[Any],
    *,
    scope: RetrievalScope | dict | None = None,
    required_refs: list[str] | None = None,
    forbidden_refs: list[str] | None = None,
    answerability: str = "answerable",
    allow_candidate: bool = False,
) -> AnswerGateResult:
    """Validate evidence before an answer is generated.

    ``answerability=absent`` and ``conflicting`` are explicit task labels from
    the eval set.  With no explicit scope the function remains compatible with
    legacy callers, while a bound request scope is used automatically.
    """

    request_scope = RetrievalScope.from_value(scope) if scope is not None else current_retrieval_scope()
    reasons: list[str] = []
    label = str(answerability or "answerable").strip().lower()
    if label in {"absent", "unanswerable", "unknown"}:
        return AnswerGateResult("insufficient", reasons=("answerability_absent",))
    if label == "insufficient":
        return AnswerGateResult("insufficient", reasons=("answerability_insufficient",))
    if label == "conflicting":
        return AnswerGateResult("conflicting", reasons=("conflicting_evidence",))

    candidates: list[str] = []
    allowed: list[str] = []
    forbidden = [str(ref).strip() for ref in (forbidden_refs or []) if str(ref).strip()]
    required = [str(ref).strip() for ref in (required_refs or []) if str(ref).strip()]
    for index, row in enumerate(items or []):
        ref = str(_row_value(row, "ref", "") or "").strip()
        if not ref:
            # Legacy assessor tests and offline callers may provide evidence
            # without citations.  Preserve that compatibility when no
            # boundary policy is requested; production scoped/required-ref
            # calls remain fail-closed.
            if required or forbidden or request_scope.project_id:
                reasons.append("missing_ref")
                continue
            candidates.append(f"__item_{index}")
            continue
        status = str(_row_value(row, "status", "active") or "active").strip().lower()
        if status in {"candidate", "quarantine", "superseded", "archived", "revoked", "expired", "conflict"} \
                and not allow_candidate:
            reasons.append(f"inactive:{status}")
            continue
        item_project = str(_row_value(row, "project_id", "") or "").strip()
        if request_scope.project_id and item_project and item_project not in {request_scope.project_id, "default"}:
            reasons.append("scope_denied")
            continue
        if any(_ref_matches(blocked, ref) for blocked in forbidden):
            reasons.append("forbidden_ref")
            continue
        candidates.append(ref)
        allowed.append(ref)

    if not candidates:
        return AnswerGateResult("insufficient", reasons=tuple(dict.fromkeys(reasons or ["no_usable_evidence"])))
    if required and not all(any(_ref_matches(expected, actual) for actual in candidates)
                            for expected in required):
        reasons.append("required_ref_missing")
        return AnswerGateResult("insufficient", tuple(allowed), tuple(dict.fromkeys(reasons)))
    return AnswerGateResult("answerable", tuple(allowed), tuple(dict.fromkeys(reasons)))


class RAGAssessor:
    def __init__(self, llm: LLMProvider, temperature: float = 0.2):
        self._llm = llm
        self._temperature = temperature
        self.calls = 0

    async def assess(self, query: str, items: list[RecallItem],
                     refs: list[str] | None = None,
                     *, scope: RetrievalScope | dict | None = None,
                     required_refs: list[str] | None = None,
                     forbidden_refs: list[str] | None = None,
                     answerability: str = "answerable") -> Sufficiency:
        """执行充分性评估：渲染 rag-assess-user.st → 结构化二次校验 → Sufficiency。"""
        gate = evaluate_answer_gate(
            query, items, scope=scope, required_refs=required_refs,
            forbidden_refs=forbidden_refs, answerability=answerability,
        )
        if not gate.allowed:
            message = "；".join(gate.reasons) or "证据不足，拒绝直接作答"
            return Sufficiency(sufficient=False, action="insufficient",
                               message=message, refs=[], answerability=gate.decision)
        if not items:
            return Sufficiency(sufficient=False, action="insufficient",
                               message="暂无检索结果，无法作答", refs=refs or [],
                               answerability="absent")
        prompt = load_prompt(
            "rag-assess-user", query=query, items=_item_block(items),
            refs=json.dumps(refs or [], ensure_ascii=False),
        )
        self.calls += 1
        resp = await self._llm.chat(
            [Message(role="user", content=prompt)], tools=None, temperature=self._temperature
        )
        output = resp.content
        if not ensure_structured(output, [_schema_ok]):
            raise AgentError("AGENT_GUARDRAIL", "充分性评估输出未通过结构化校验")
        obj = obj_from(output)
        candidate_refs = {str(item.ref).strip() for item in items if str(item.ref).strip()}
        output_refs = [str(ref).strip() for ref in (obj.get("refs") or [])]
        if output_refs and candidate_refs and not all(
            _ref_allowed(ref, candidate_refs) for ref in output_refs
        ):
            return Sufficiency(
                sufficient=False,
                action="insufficient",
                message="评估结果引用了本次检索之外的来源，拒绝直接作答",
                refs=[],
                answerability="scope_denied",
            )
        post_gate = evaluate_answer_gate(
            query, items, scope=scope, required_refs=required_refs,
            forbidden_refs=forbidden_refs, answerability=answerability,
        )
        if not post_gate.allowed:
            return Sufficiency(sufficient=False, action="insufficient",
                               message="生成前后证据闸门未通过：" + "；".join(post_gate.reasons),
                               refs=[], answerability=post_gate.decision)
        return Sufficiency(
            sufficient=bool(obj["sufficient"]),
            action=str(obj["action"]),
            reformulated_query=str(obj.get("reformulated_query") or ""),
            message=str(obj.get("message") or ""),
            refs=obj.get("refs") or refs or [],
            answerability=str(obj.get("answerability") or "answerable"),
        )


def obj_from(output: str) -> dict:
    """供测试复用的 JSON 抽取（走 guardrails.extract_json 保证剥壳/配对）。"""
    from agentlab.core.guardrails import extract_json
    return extract_json(output)
