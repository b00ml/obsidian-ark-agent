"""Export versioned public JSON schemas for HTTP/SSE and MCP contracts."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentlab"))
sys.path.insert(0, str(ROOT))

from agentlab.contracts import Artifact, Citation, RetrievalItem, Session, SourceDocument  # noqa: E402
from agentlab.runtime.serve_contract import ResponsesRequest  # noqa: E402
from obsidian_agent_brain.tool_registry import BRAIN_TOOL_SPECS, validate_specs  # noqa: E402


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "responses-request-v1.json": ResponsesRequest.model_json_schema(),
        "retrieval-item-v1.json": RetrievalItem.model_json_schema(),
        "source-document-v1.json": SourceDocument.model_json_schema(),
        "citation-v1.json": Citation.model_json_schema(),
        "session-v1.json": Session.model_json_schema(),
        "artifact-v1.json": Artifact.model_json_schema(),
    }
    for name, schema in models.items():
        (out_dir / name).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    errors = validate_specs()
    if errors:
        raise SystemExit("MCP contract errors: " + "; ".join(errors))
    manifest = {
        "schema": "obsidian-brain-mcp-tools-v1",
        "tools": [
            {"name": spec.name, "module": spec.module_suffix,
             "permission": spec.permission, "timeout": spec.timeout,
             "side_effects": spec.side_effects, "idempotent": spec.idempotent,
             "requires_approval": spec.requires_approval}
            for spec in BRAIN_TOOL_SPECS
        ],
    }
    (out_dir / "mcp-tools-v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    openapi = {
        "openapi": "3.0.3", "info": {"title": "agentlab serve", "version": "v1"},
        "paths": {
            "/v1/responses": {"post": {"summary": "Run an agent response",
                "requestBody": {"required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/ResponsesRequest"}}}},
                "responses": {"200": {"description": "SSE response stream"},
                               "400": {"description": "Invalid request"},
                               "401": {"description": "Unauthorized"}}}},
            "/health": {"get": {"summary": "Health check", "responses": {"200": {"description": "Healthy"}}}},
        },
        "components": {"schemas": {"ResponsesRequest": ResponsesRequest.model_json_schema()}},
        "x-sse-events": ["response.output_text.delta", "response.output_item.added",
                          "response.output_item.done", "response.completed", "response.failed"],
    }
    (out_dir / "agentlab-openapi-v1.json").write_text(
        json.dumps(openapi, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    target = ROOT / "docs_public" / "contracts"
    if check_only:
        with tempfile.TemporaryDirectory() as tmp:
            export(Path(tmp))
            expected = {p.name: p.read_bytes() for p in Path(tmp).glob("*.json")}
        actual = {p.name: p.read_bytes() for p in target.glob("*.json")} if target.exists() else {}
        missing = sorted(set(expected) - set(actual))
        stale = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        extra = sorted(set(actual) - set(expected))
        if missing or stale or extra:
            raise SystemExit(f"contract schemas stale: missing={missing} stale={stale} extra={extra}")
        print("contract schemas up to date")
    else:
        export(target)
        print("exported docs_public/contracts")
