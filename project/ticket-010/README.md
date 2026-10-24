# Ticket 010: Add System One and Jev-compatible typed decision layer for DSL and DockURI

- **ID**: ticket-010
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-24

## Goal and scope

Add a normative System One / Jev-compatible typed decision layer (`standard/system_one_decision.py` and `schemas/system-one-decision.schema.json`) to `wellmanifest/nl-dsl-llm` covering all 10 open-source Jev / System One model families (`Clef`/`Clef-flash`, `GLiNER2.5-Decide`, `Laya`, `Kev`, `Laya-MLX`, `SemIf`, `NanoJev`, `Bespoke-Nimble`, `jevlike`, `jevos`), single-pass `choice`/`score`/`noul` decision evaluation over `OptionNode`, `DSLCommand`, and DockURI `proc://` candidates, and `policy-dsl` cross-question constraints.

## Acceptance criteria

- [x] AC-01: `schemas/system-one-decision.schema.json` defines the `wellmanifest.system-one-decision/v1` JSON Schema with tuple-bound `choice`/`score`/`noul` questions and normalized probability distributions.
- [x] AC-02: `standard/system_one_decision.py` implements the SSOT registry of all 10 Jev/System One model families, `select_decision_profile`, `SystemOneDecisionEngine`, `CrossQuestionConstraint`, and `validate_system_one_decision_envelope`.
- [x] AC-03: `standard/test_system_one_decision.py` verifies all 10 model families, cross-question constraints, envelope validation, and `NLDSLLLMBridge` / `OptionNetworkEngine` integration; `pytest` and `./project/governance-check.sh` pass with 0 errors.

## SESSION_EXECUTION_AUTHORIZATION

User authorized autonomous execution across `~/github/*/*` to implement LLM / System One / Jev-compatible DockURI URI process decisions conforming to `wellmanifest/nohardcode`, `wellmanifest/ssot`, `wellmanifest/dsl`, `wellmanifest/policy-dsl`, and `wellmanifest/nl-*`.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
