# Ticket 015: Harden RFC6455 WebSocket handshake/control frames, JSON depth/NaN bounds, and System One contract gates

- **ID**: ticket-015
- **Owner**: unresolved:human
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-11-11

## Goal and scope

Extract RFC 6455 WebSocket handshake/framing into `standard/nl_dsl_ws_framing.py` and System One schema/formatting helpers into `standard/system_one_profiles.py` so every module stays comfortably below 500 lines, while hardening RFC 6455 `Sec-WebSocket-Key` / `Upgrade: websocket` validation, control-frame (`<= 125` bytes, `FIN=1`), `RSV=0`, and minimal payload length checks, `NLStreamFrame` `NaN`/`Infinity`/`MAX_JSON_DEPTH = 64` bounds, and `select_operation` / `build_selection_question` contract & `:252` digest gates.

## Acceptance criteria

- [x] AC-01: Invalid or non-ASCII `Sec-WebSocket-Key` upgrades return HTTP 400 without `UnicodeEncodeError`, `encode_ws_frame` / `decode_ws_frame` reject `> 125`-byte or fragmented control frames, non-zero `RSV` bits, and non-minimal extended lengths, and `NLStreamFrame` rejects `NaN`/`Infinity` and `> 64` JSON nesting.
- [x] AC-02: `build_selection_question` and `select_operation` align `:252` candidate slicing with `registry_digest` and fail closed on `bool`/non-int `risk_level`, non-mapping `arguments`, and non-string `decision_provider` fields without raising `TypeError`/`ValueError`.
- [x] AC-03: All repository Python modules stay strictly below 500 lines and pass `pytest`, `python3 standard/nl_dsl_llm.py --self-test`, `ruff check`, `./project/governance-check.sh --base origin/main`, and `wellman check`.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
