# Ticket 004: WebSocket streaming protocol and live Option Network autocomplete

- **ID**: ticket-004
- **Owner**: agent:gemini
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-09-27

## Goal and scope

Implement WebSocket real-time bidirectional streaming protocol for chunked audio and live Option Network DAG autocomplete in `wellmanifest/nl-dsl-llm`.

SESSION_EXECUTION_AUTHORIZATION: User requested sequential execution of ecosystem backlog tasks ("kolejno", "kontynuuj"), prioritizing the real-time WebSocket streaming bridge and Option Network autocomplete protocol for voice and interactive terminal/web consumers.

Scope includes:
1. Normative specification update in `spec/NL_DSL_LLM_SPECIFICATION.md` adding Section 3.5 (WebSocket Streaming & Autocomplete Protocol) and Section 3.6 (Option Network Live Autocomplete Engine).
2. Reference WebSocket server in `standard/nl_dsl_llm_ws.py` implementing bidirectional framing (`AUDIO_CHUNK`, `TRANSCRIPTION_PARTIAL`, `OPTION_NETWORK_UPDATE`, `INTENT_MATCH`, `EXECUTE_DSL`).
3. Streaming Option Network autocomplete engine delivering sub-50ms DAG projection updates as partial tokens/audio arrive.
4. Conformance tests in `standard/test_websocket_streaming.py` validating frame parsing, autocomplete DAG streaming, and session lifecycle.
5. End-to-end example client in `examples/websocket_streaming_client.py` demonstrating live voice chunk simulation and real-time Option Network autocomplete.

## Acceptance criteria

- [x] AC-01: WebSocket streaming protocol specification defined in `spec/NL_DSL_LLM_SPECIFICATION.md` Section 3.5 & 3.6.
- [x] AC-02: Reference WebSocket server in `standard/nl_dsl_llm_ws.py` supporting bidirectional streaming of audio chunks, partial transcripts, and live Option Network DAG suggestions.
- [x] AC-03: Real-time latency benchmark demonstrating sub-50ms autocomplete generation from streaming prefix tokens.
- [x] AC-04: Conformance test suite in `standard/test_websocket_streaming.py` passes all tests.
- [x] AC-05: `./project/governance-check.sh` passes with 0 errors and 0 warnings.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
