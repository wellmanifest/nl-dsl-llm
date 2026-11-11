# Ticket 014: Modularize NL-DSL-LLM standard modules and harden WS framing, DSL quoting, and System One ACL/NaN validation

- **ID**: ticket-014
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-11-11

## Goal and scope

Decompose the three oversized Python modules in `standard/` (`standard/nl_dsl_llm.py` 915L, `standard/system_one_decision.py` 674L, and `standard/nl_dsl_llm_ws.py` 516L) so every module in `wellmanifest/nl-dsl-llm` stays strictly below 500 lines while preserving 100% public re-exports and CLI entrypoints, and fix concrete security/semantic bugs across DSL parsing, WebSocket framing, and System One safety gates:
- Parse quoted whitespace-containing arguments accurately in `DSLCommand.from_text`, reject malformed non-`key=val` tokens, guard `SemanticCache.match` against non-string slot values, and validate `InterfaceAdapters.handle_mcp_tool_call` argument types.
- Bound WebSocket handshake headers (`16 KiB`) and frame payloads (`1 MiB`) in `standard/nl_dsl_llm_ws.py`, vectorize XOR frame masking, eliminate millisecond `session_id` collisions, validate `NLStreamFrame.from_json` types, and emit `STREAM_ERROR` on malformed frames.
- URL-encode `bound_arguments` in `standard/system_one_decision.py` Action URI projection, cache JSON schema loading, reject `NaN`/`Infinity`/`bool`-as-`int`/`int`-as-`bool` values in `validate_system_one_decision_envelope`, and fix the `select_operation` empty-`allowed_uris=[]` ACL bypass.

## Acceptance criteria

- [x] AC-01: Every Python module in `standard/` and repository-wide is strictly `< 500` lines, with full public re-export compatibility.
- [x] AC-02: Adversarial tests in `standard/test_hardening.py` cover quoted DSL arguments, WS payload/handshake bounds, vectorized masking, `session_id` uniqueness, URL-encoded Action URIs, `NaN`/`bool` envelope rejection, and empty-ACL denial.
- [x] AC-03: `pytest`, `ruff`, `./project/governance-check.sh`, and `wellman check` pass with 0 errors.

## SESSION_EXECUTION_AUTHORIZATION

User authorized continuous autonomous bug-fixing, modularization, optimization, testing, and Validator-gated merging across `wellmanifest/*` repositories.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
