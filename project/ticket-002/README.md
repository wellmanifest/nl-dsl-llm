# Ticket 002: gRPC transport parity, 4-tier pipeline, and paxlet adoption case study

- **ID**: ticket-002
- **Owner**: human:tom
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-09-26

## Goal and scope

Extend the `wellmanifest/nl-dsl-llm` standard specification and reference implementation to incorporate a 4-tier architecture (introducing Layer 1.5 Semantic Cache with strict Polarity & Slot Guards), standardize gRPC IPC transport parity alongside CLI, REST, and MCP, and document the runtime daemon adoption in `paxlet-com/nl-dsl-llm`.

## Acceptance criteria

- [x] AC-01: Update normative specification `spec/NL_DSL_LLM_SPECIFICATION.md` detailing the 4-tier pipeline and 4-interface parity (CLI, REST, MCP, gRPC).
- [x] AC-02: Add Protobuf v1 definition `proto/wellmanifest/nl_dsl_llm/v1/runtime.proto` defining `NLRuntimeService` for high-performance IPC.
- [x] AC-03: Extend reference standard `standard/nl_dsl_llm.py` and pytest suite `standard/test_conformance.py` with `SemanticCache` (Layer 1.5) and polarity guard verification.
- [x] AC-04: Document runtime adoption in `examples/paxlet_runtime_adoption.md`.
- [x] AC-05: Conformance self-tests (7/7) and governance checks pass (`./project/governance-check.sh`).

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
