# agentlab

agentlab is a small, reusable Python Agent runtime built around
Agent = LLM + Tools + Loop.

The package keeps the execution loop independent from the UI and business
domain. Tools, memory, retrieval, approvals, tracing, and HTTP/SSE serving
are assembled through explicit adapters.

## Highlights

- event-driven Agent Loop with tool batches, cancellation, steering, and
  context budgeting;
- registry-based tools with read/write/danger permissions and optional HITL
  approval;
- Markdown-backed long-term memory under an Obsidian vault;
- keyword and optional vector retrieval with bounded synchronization;
- token-aware tracing, resilience, and an HTTP/SSE service compatible with
  the Ark plugin;
- unit tests that run without network access or real credentials.

## Quick start

~~~bash
cd agentlab
python -m pip install -e ".[dev]"
Copy-Item config/config.example.json config/config.json
# Edit config/config.json and set vault_root and the model credentials.
python -m agentlab.runtime.cli tools list
python -m agentlab.runtime.cli run "Explain the available tools"
~~~

The LLM API key can also be supplied through AGENT_LLM_API_KEY. Keep
config.json local; only the example configuration belongs in version
control.

## Tests

~~~bash
python -m unittest discover -s tests -v
~~~

## Package layout

~~~text
agentlab/
  core/       Agent loop, messages, context, routing, resilience
  tools/      Tool registry, connectors, retrieval and demo tools
  memory/     Memory storage, recall, consolidation and session ranges
  rag/        Retrieval and vector index integration
  runtime/    CLI, configuration, tracing and HTTP/SSE service
  eval/       Offline behavior and quality evaluation helpers
config/       Local configuration templates
tests/        Unit tests
~~~

## Memory model

Long-term memories are ordinary Markdown files under
<vault>/ark/memory/. They can be inspected and edited in Obsidian, and are
organized into core, context, procedures, decisions, sessions, and archive
categories. The migration helper
agentlab/scripts/migrate_memory_to_markdown.py can import legacy SQLite
memory data into this layout.
