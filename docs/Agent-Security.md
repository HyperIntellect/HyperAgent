# Agent Security

## Overview

This document defines the security model for HyperAgent with a focus on prompt injection resistance.
It covers threat boundaries, implemented controls, configuration, testing, and operational response.

Primary goals:

- Prevent untrusted content from overriding agent policy or system behavior.
- Limit damage if model behavior is manipulated.
- Detect and block unsafe tool invocations.
- Keep defenses testable and auditable.

## Threat Model

### In-Scope Threats

- Direct jailbreak attempts in user messages.
- Prompt injection embedded in:
  - web content
  - tool outputs
  - attached files (OCR/text extraction)
  - compressed context summaries
- Instruction laundering where untrusted content gets elevated into system-level context.
- Unsafe tool execution caused by manipulated model output.
- Data exfiltration attempts via URL or shell/code patterns.

### Out-of-Scope Threats

- Compromise of external infrastructure (DB/host/cloud account takeover).
- Credential leaks caused outside this repository.
- Adversarial risks requiring model-provider side controls only.

## Trust Boundaries

Trust assumptions in this system:

- Trusted:
  - static system prompts and server-side policy code
  - explicit backend state transitions and tool contracts
- Untrusted:
  - user input
  - attached file text
  - tool outputs (including local code/shell output)
  - web search snippets and fetched page content
  - generated summaries of historical content

Critical rule: untrusted content is treated as data, not as instructions.

## Defense-in-Depth Controls

### 1) Input Guardrails (Entry Point)

- `input_scanner` checks requests for prompt-injection/jailbreak patterns.
- Enforced on both streaming and non-streaming query endpoints.

Key files:

- `backend/app/api/query.py`
- `backend/app/guardrails/scanners/input_scanner.py`

### 2) Attachment Context Hardening

- Extracted file text is sanitized by a dedicated untrusted-content scanner (`high` sensitivity).
- Attachment context is not appended to `SystemMessage`.
- Attachment context is injected as a `HumanMessage` with explicit "untrusted data" framing.

Key files:

- `backend/app/api/query_helpers.py`
- `backend/app/agents/state.py`
- `backend/app/agents/supervisor.py`
- `backend/app/agents/subagents/task.py`

### 3) Tool Argument Guardrails

- Tool args are scanned before execution.
- URL checks block local/internal/private targets and unsafe schemes.
- Code checks block known destructive patterns.

Key files:

- `backend/app/guardrails/scanners/tool_scanner.py`
- `backend/app/agents/tools/tool_pipeline.py`

### 4) Tool Output Prompt-Injection Sanitization

- Tool results are scanned as untrusted content before reinsertion into model context.
- Per-tool sensitivity levels (`low` / `medium` / `high`) are applied.
- Sanitized content is wrapped as untrusted data when patterns are detected.

Key files:

- `backend/app/agents/tools/tool_pipeline.py`
- `backend/app/guardrails/scanners/untrusted_content_scanner.py`

### 5) Context Compression Hardening

- Compression prompt explicitly forbids preserving jailbreak/policy-bypass instructions.
- Reinjected summaries are sanitized.
- Summaries are injected as non-authoritative `HumanMessage` context, not as system policy.

Key files:

- `backend/app/agents/context_compression.py`

### 6) Output Guardrails

- Model outputs are scanned for toxicity/PII/harmful content before final emission paths.

Key files:

- `backend/app/guardrails/scanners/output_scanner.py`
- `backend/app/agents/subagents/task.py`

### 7) Policy Engine + HITL

- Tool capabilities are contract-defined and risk-scored.
- High-risk actions can require explicit user approval.
- Deny/require-approval decisions are enforced before tool execution.

Key files:

- `backend/app/agents/policy/engine.py`
- `backend/app/agents/tools/registry.py`
- `backend/app/agents/hitl/tool_risk.py`
- `backend/app/agents/tools/tool_pipeline.py`

## Dedicated Untrusted Content Scanner

Scanner: `backend/app/guardrails/scanners/untrusted_content_scanner.py`

Capabilities:

- Detects instruction-like patterns in untrusted intermediate content.
- Returns sanitized content via `ScanResult.sanitized_content`.
- Supports sensitivity levels:
  - `low`: lightweight filtering
  - `medium`: balanced defaults
  - `high`: strict filtering for web/file/summary contexts
- Provides async `scan()` and sync `sanitize_untrusted_content()` helper.

Recommended sensitivity:

- `high`: `web_search`, `web_extract_structured`, `http_request`, `browser_dom_query`, `browser_get_accessibility_tree`, `file_read`, attachment text, context summaries.
- `medium`: `execute_code`, `shell_exec` outputs.
- `low`: only when false positive risk is clearly unacceptable and content cannot drive tool decisions.

## Configuration

Security-relevant settings in `backend/app/config.py`:

- `guardrails_enabled`
- `guardrails_input_enabled`
- `guardrails_output_enabled`
- `guardrails_tool_enabled`
- `guardrails_violation_action` (`block` recommended)
- `hitl_enabled`
- `hitl_default_risk_threshold`

Production recommendation:

- Keep all guardrails enabled.
- Keep violation action as `block`.
- Keep HITL enabled for high-risk tools.

## Security Tests

Primary regression coverage:

- `backend/tests/test_prompt_injection_hardening.py`
- `backend/tests/test_untrusted_content_scanner.py`
- `backend/tests/test_guardrails.py`
- `backend/tests/test_security_controls.py`

Run focused checks:

```bash
cd backend
uv run pytest tests/test_prompt_injection_hardening.py tests/test_untrusted_content_scanner.py -q
```

Run full backend suite:

```bash
cd backend
uv run pytest -q
```

## Operational Response Playbook

If prompt-injection behavior is observed:

1. Capture request ID/task ID and offending content source (user/file/tool/web/summary).
2. Confirm which scanner path processed it (input/tool/untrusted/output).
3. Inspect run ledger + events for tool calls and policy decisions.
4. Raise scanner sensitivity for affected source path.
5. Add a regression test reproducing the bypass before shipping the fix.
6. Review tool contracts and HITL requirements for impacted tools.

## Secure Development Checklist

For any new feature/tool that introduces model context:

- Classify content as trusted or untrusted.
- Never place untrusted text in system prompt or system-level policy channels.
- Pass untrusted content through `untrusted_content_scanner`.
- Add/adjust per-tool sensitivity mapping if tool outputs feed the model.
- Ensure tool args are validated/scanned before execution.
- Add regression tests for prompt injection and policy bypass attempts.
- Verify behavior in both streaming and non-streaming API paths.

## Known Limitations

- Pattern-driven scanners can have false positives/false negatives.
- Some paraphrased adversarial instructions may evade static pattern checks.
- Defense relies on layered controls; no single scanner is sufficient.

Mitigation strategy:

- Keep layered controls enabled (input + untrusted-content + tool-arg + policy + HITL).
- Continuously expand test corpus with real bypass examples.

